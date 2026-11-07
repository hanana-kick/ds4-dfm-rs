//! V4.1 rolling requests with packed dense/MoE MTP target verification.
//!
//! Entrpi v0.6.5 supplies the ownership rule: one GPU owner admits/retires
//! slots between rounds. Each request owns its draft towers, KV, Engram feed,
//! image features, RNG and accepted-token history. The ordinary host stepper
//! checks every accepted target token, including DSML syntax and stop rules.
use std::cell::RefCell;
use std::io::Write;
use std::time::Instant;

use ds4_core::v41_media::V41Images;
use ds4_core::{Model, V41Batch, V41Request, V41Sampling};
use ds4_kv::Store as KvStore;

use crate::dsml::SampleOverride;
use crate::generate::{GenerateError, GenerateOutcome, V41ServeRoute};
use crate::parse::{ChatPart, ParsedRequest};
use crate::route::decode_budget;
use crate::serve_cont::{
    cont_prompt_tokens, ContExec, ContOwnedResult, ContOwnedWork, ContPreparedPrompt, ContProbe,
    ContSource, ContStepper,
};
use crate::stream::ReqTimings;
use crate::tool_memory::{ToolMemory, ToolMemoryConfig};

pub struct V41ContLane<'m> {
    model: &'m Model,
    route: V41ServeRoute,
    batch: V41Batch<'m>,
    memory: RefCell<ToolMemory>,
    ctx: i32,
    stop_requested: Option<fn() -> bool>,
    steps: u64,
    max_rows: usize,
}

struct Prepared {
    prompt: Vec<u8>,
    tokens: Vec<i32>,
    images: Option<V41Images>,
    budget: usize,
}
struct State<'m> {
    native: V41Request<'m>,
    stepper: ContStepper,
    sampling: V41Sampling,
    rng: u64,
    pending: Option<i32>,
    done: bool,
    eos: bool,
    t_admit: Instant,
    t_first: Option<Instant>,
    t_arrive: Instant,
    prefill_ms: f64,
    steps: i32,
}
struct Live<'m> {
    work: ContOwnedWork,
    state: State<'m>,
}

fn engine_error(e: impl std::fmt::Display) -> GenerateError {
    GenerateError::Engine(e.to_string())
}
fn push(out: &mut dyn Write, bytes: &[u8]) -> Result<(), GenerateError> {
    out.write_all(bytes)
        .and_then(|()| out.flush())
        .map_err(|_| GenerateError::Io)
}
fn sampling(p: &ParsedRequest) -> V41Sampling {
    V41Sampling {
        temperature: p.temperature,
        top_p: p.top_p,
        min_p: p.min_p,
        top_k: p.top_k,
        seed: p.seed,
        freq_penalty: p.frequency_penalty,
        presence_penalty: p.presence_penalty,
        dry_multiplier: if p.dry_set { p.dry_multiplier } else { 0. },
        dry_base: if p.dry_set && p.dry_base > 1. {
            p.dry_base
        } else {
            1.75
        },
        dry_allowed_length: if p.dry_set && p.dry_allowed_length > 0 {
            p.dry_allowed_length
        } else {
            2
        },
    }
}

impl<'m> V41ContLane<'m> {
    pub fn new(
        model: &'m Model,
        route: V41ServeRoute,
        width: usize,
        floor_bytes: u64,
        tool_memory: ToolMemoryConfig,
    ) -> Result<Self, GenerateError> {
        if !route.no_kv_reuse {
            return Err(GenerateError::BadRequest(
                "V4.1 concurrent lane requires --prefix-reuse off",
            ));
        }
        let ctx = model
            .v41_ctx()
            .ok_or_else(|| engine_error("V4.1 context metadata is absent"))?;
        let batch = V41Batch::new(
            model,
            &route.model_path,
            route.engram_dir.as_deref(),
            route.no_engram,
            route.zchain.as_deref(),
            route.posttrain.as_deref(),
            width,
            floor_bytes,
        )
        .and_then(|batch| batch.with_mtp(
            route.dspark.unwrap_or(1), route.verify_k, route.graph.unwrap_or(true),
        ))
        .map_err(engine_error)?;
        Ok(Self {
            model,
            route,
            batch,
            memory: RefCell::new(ToolMemory::from_config(tool_memory)),
            ctx: ctx as i32,
            stop_requested: None,
            steps: 0,
            max_rows: 0,
        })
    }

    fn prepare(
        &self,
        p: &ParsedRequest,
        prepared: Option<&ContPreparedPrompt>,
        default_tokens: i32,
    ) -> Result<Prepared, GenerateError> {
        let (prompt, mut tokens) = match prepared {
            Some(v) => (v.prompt.clone(), v.tokens.clone()),
            None => {
                let mut p = p.clone();
                cont_prompt_tokens(self, &mut p)?
            }
        };
        // Routing uses unexpanded text tokens. Expand exactly once at admission,
        // before pricing/allocating KV or emitting any SSE headers. Pixels are
        // owned by this request, never cached in a process-global mutable slot.
        let images = if p.images.is_empty() {
            None
        } else {
            let meta = self.route.vision.as_ref().ok_or(GenerateError::BadRequest(
                "image input requires --vision-gguf",
            ))?;
            let mut order = Vec::new();
            for message in &p.messages {
                for part in &message.parts {
                    if let ChatPart::Image(i) = part {
                        order.push(
                            p.images
                                .get(*i)
                                .ok_or(GenerateError::BadRequest("invalid image block index"))?
                                .data
                                .as_ref(),
                        );
                    }
                }
            }
            let images =
                V41Images::expand(&tokens, &order, meta, self.ctx as usize).map_err(|_| {
                    GenerateError::BadRequest(
                        "invalid V4.1 image request or expanded context/memory budget exceeded",
                    )
                })?;
            tokens = images.tokens.clone();
            Some(images)
        };
        if tokens.is_empty() || tokens.len() >= self.ctx as usize {
            return Err(GenerateError::BadRequest(
                "V4.1 prompt leaves no room in model context",
            ));
        }
        let budget = decode_budget(p.max_tokens_set, p.max_tokens, default_tokens).max(0) as usize;
        let budget = budget.min(self.ctx as usize - tokens.len());
        Ok(Prepared {
            prompt,
            tokens,
            images,
            budget,
        })
    }

    fn start(
        &self,
        prepared: Prepared,
        p: &ParsedRequest,
        id: &str,
        created: i64,
        cors: bool,
        defaults: i32,
        arrived: Instant,
        out: &mut dyn Write,
    ) -> Result<State<'m>, GenerateError> {
        // A zero output request still carries exact expanded usage but does no
        // GPU prefill. It is handled separately, before creating a native state.
        let mut parsed = p.clone();
        crate::generate::prepare_required_prefixes(
            &mut parsed,
            crate::render::syntax_for_model_id(self.model.model_id()),
            |literal| Ok(self.model.vocab().encode_rendered_bytes(literal)),
        )?;
        let native = self
            .batch
            .request(
                prepared.tokens.clone(),
                prepared.budget.max(1),
                prepared.images,
            )
            .map_err(engine_error)?;
        let (mut stepper, head) = ContStepper::new(
            &parsed,
            self.model.model_id(),
            id,
            created,
            cors,
            defaults,
            prepared.prompt,
            prepared.tokens.len() as i32,
            self.ctx,
        );
        let head = stepper.admitted_head(head, 0, prepared.tokens.len() as i32);
        push(out, &head)?;
        Ok(State {
            native,
            stepper,
            sampling: sampling(p),
            rng: p.seed,
            pending: None,
            done: prepared.budget == 0,
            eos: false,
            t_admit: Instant::now(),
            t_first: None,
            t_arrive: arrived,
            prefill_ms: 0.,
            steps: 0,
        })
    }

    fn zero(
        &self,
        prepared: Prepared,
        work: &mut ContOwnedWork,
    ) -> Result<GenerateOutcome, GenerateError> {
        let n = prepared.tokens.len() as i32;
        let (mut stepper, head) = ContStepper::new(
            &work.parsed,
            self.model.model_id(),
            &work.job_id,
            work.created,
            work.cors,
            work.default_tokens,
            prepared.prompt,
            n,
            self.ctx,
        );
        let head = stepper.admitted_head(head, 0, n);
        push(work.out.as_mut(), &head)?;
        let (tail, result) = stepper.finalize(false, 0, n, ReqTimings::default(), work.cors);
        push(work.out.as_mut(), &tail)?;
        Ok(result)
    }

    fn sample_emit(&mut self, state: &mut State<'m>, out: &mut dyn Write) -> Result<(), GenerateError> {
        if state.done { return Ok(()); }
        loop {
            // The target sample is authoritative. A deterministic draft is
            // accepted only on equality, preserving the requested distribution
            // without sampling rejected future rows or advancing their RNG.
            let excluded = state.stepper.sample_exclude(self.model.token_eos());
            let token = match state.stepper.sample_override() {
                SampleOverride::Token(t) if t == excluded => {
                    return Err(engine_error("forced token conflicts with EOS policy"))
                }
                SampleOverride::Token(t) => t,
                SampleOverride::None => state.native.sample(
                    state.sampling, &mut state.rng,
                    state.stepper.v41_syntax_greedy(), excluded,
                ).map_err(engine_error)?,
            };
            state.t_first.get_or_insert_with(Instant::now);
            if self.model.token_is_stop(token) {
                state.stepper.mark_stop();
                state.eos = true;
                state.done = true;
                state.pending = None;
            } else {
                let piece = self.model.token_text(token).map_err(engine_error)?;
                let step = state.stepper.feed(&piece);
                state.native.record(token);
                push(out, &step.bytes)?;
                state.done = step.done;
                // EOS, output limits, stop strings and DSML tool boundaries
                // win BEFORE accepting another already-computed draft row.
                if !state.done && state.native.advance_verified(token).map_err(engine_error)? {
                    continue;
                }
                state.pending = (!state.done).then_some(token);
            }
            state.native.finish_round().map_err(|error| {
                self.batch.invalidate();
                engine_error(error)
            })?;
            return Ok(());
        }
    }

    fn finish(&self, state: &mut State<'m>, cors: bool) -> (Vec<u8>, GenerateOutcome) {
        let n = state.stepper.prompt_n;
        let t = ReqTimings {
            valid: true,
            ttft_ms: state.t_first.map_or(0., |t| {
                t.duration_since(state.t_arrive).as_secs_f64() * 1000.
            }),
            prefill_ms: state.prefill_ms,
            decode_ms: state
                .t_first
                .map_or(0., |t| t.elapsed().as_secs_f64() * 1000.),
            prefill_tokens: n,
            prefill_cached: 0,
            decode_tokens: state.stepper.completion(),
            decode_steps: state.steps,
        };
        let (tail, mut result) = state.stepper.finalize(state.eos, 0, n, t, cors);
        let (rounds, offered, accepted) = state.native.spec_stats();
        result.speculation_active = offered != 0;
        if offered != 0 {
            eprintln!("[v41-mtp] request complete rounds={rounds} offered={offered} accepted={accepted}");
        }
        if let Some((calls, raw)) = state.stepper.take_tool_replay() {
            self.memory.borrow_mut().remember(&calls, &raw);
        }
        (tail, result)
    }

    fn report_step(&mut self, n: usize) {
        self.steps += 1;
        if n > self.max_rows {
            self.max_rows = n;
            eprintln!("[v41-batch] packed_decode rows={n} steps={} verify_rows={} (shared dense/MoE; request-local MTP/KV/Engram)", self.steps, self.batch.last_verified_rows());
        }
    }

    fn settle(
        source: &mut dyn ContSource,
        results: &mut Vec<ContOwnedResult>,
        key: usize,
        result: Result<GenerateOutcome, GenerateError>,
    ) {
        if let Some(result) = source.settle_owned(ContOwnedResult { key, result }) {
            results.push(result);
        }
    }

    fn abort(&self, mut job: Live<'m>, error: GenerateError) -> ContOwnedResult {
        let error = if job.work.parsed.stream && !matches!(error, GenerateError::Io) {
            let text = error.to_string();
            let bytes = job.state.stepper.fail(&text);
            match push(job.work.out.as_mut(), &bytes) {
                Ok(()) => GenerateError::Streamed(text),
                Err(e) => e,
            }
        } else {
            error
        };
        // Destructors drain Engram I/O and synchronize before freeing this KV.
        ContOwnedResult {
            key: job.work.key,
            result: Err(error),
        }
    }
}

impl ContProbe for V41ContLane<'_> {
    fn prompt_tokens(&self, p: &mut ParsedRequest) -> Result<(Vec<u8>, Vec<i32>), GenerateError> {
        cont_prompt_tokens(self, p)
    }
    fn seq_cap(&self) -> i32 {
        self.ctx
    }
    fn bank_live(&self, _bank: i32) -> Option<(u64, i32)> {
        None
    }
}
impl<'m> ContExec for V41ContLane<'m> {
    fn model_id(&self) -> i32 {
        self.model.model_id()
    }
    fn chat_system_token(&self) -> &'static str {
        self.model.vocab().chat_system_token()
    }
    fn render_request(&self, p: &ParsedRequest) -> Result<Vec<u8>, GenerateError> {
        crate::chat_input::render_model(
            self.model.chat_template(),
            self.model.model_id(),
            p,
            self.chat_system_token(),
        )
    }
    fn restore_chat(&self, p: &mut ParsedRequest) -> Result<(), GenerateError> {
        self.memory.borrow_mut().attach(&mut p.messages);
        Ok(())
    }
    fn seq_cap(&self) -> i32 {
        self.ctx
    }
    fn max_seq(&self) -> i32 {
        self.batch.width() as i32
    }
    fn set_stop_requested(&mut self, f: Option<fn() -> bool>) {
        self.stop_requested = f;
    }
    fn encode_chat(&self, p: &[u8]) -> Vec<i32> {
        self.model.vocab().encode_rendered_bytes(p)
    }
    fn encode_text(&self, p: &str) -> Vec<i32> {
        self.model.vocab().encode_text(p)
    }
    fn prepare_tokens(
        &self,
        p: &ParsedRequest,
        _prompt: &[u8],
        tokens: Vec<i32>,
    ) -> Result<Vec<i32>, GenerateError> {
        if p.return_token_ids {
            return Err(GenerateError::BadRequest(
                "V4.1 packed lane does not project return_token_ids; use the serial lane",
            ));
        }
        if !p.images.is_empty() && self.route.vision.is_none() {
            return Err(GenerateError::BadRequest(
                "image input requires --vision-gguf",
            ));
        }
        Ok(tokens)
    }

    fn generate(
        &mut self,
        p: &ParsedRequest,
        id: &str,
        created: i64,
        cors: bool,
        defaults: i32,
        arrived: Instant,
        _hold: &mut dyn FnMut(i32, Option<(u64, i32)>) -> Option<i32>,
        _store: Option<&mut KvStore>,
        out: &mut dyn Write,
    ) -> Result<GenerateOutcome, GenerateError> {
        let prepared = self.prepare(p, None, defaults)?;
        if prepared.budget == 0 {
            let n = prepared.tokens.len() as i32;
            let (mut stepper, head) = ContStepper::new(
                p,
                self.model.model_id(),
                id,
                created,
                cors,
                defaults,
                prepared.prompt,
                n,
                self.ctx,
            );
            let head = stepper.admitted_head(head, 0, n);
            push(out, &head)?;
            let (tail, result) = stepper.finalize(false, 0, n, ReqTimings::default(), cors);
            push(out, &tail)?;
            return Ok(result);
        }
        if !self
            .batch
            .can_admit(&prepared.tokens, prepared.budget, prepared.images.as_ref())
            .map_err(engine_error)?
        {
            return Err(GenerateError::Capacity(
                "V4.1 request exceeds current memory headroom; reduce prompt/output or retry",
            ));
        }
        let mut state = self.start(prepared, p, id, created, cors, defaults, arrived, out)?;
        let run = (|| {
            while !state.done {
                if self.stop_requested.is_some_and(|f| f()) {
                    return Err(engine_error("server shutdown"));
                }
                push(out, &state.stepper.heartbeat(Instant::now()))?;
                if !state.native.ready() {
                    let ready = state.native.prefill().map_err(|e| {
                        self.batch.invalidate();
                        engine_error(e)
                    })?;
                    if !ready {
                        continue;
                    }
                    state.prefill_ms = state.t_admit.elapsed().as_secs_f64() * 1000.;
                } else if let Some(token) = state.pending.take() {
                    self.batch
                        .decode(&mut [(&mut state.native, token)])
                        .map_err(engine_error)?;
                    self.report_step(1);
                    state.steps += 1;
                }
                self.sample_emit(&mut state, out)?;
            }
            let (tail, result) = self.finish(&mut state, cors);
            push(out, &tail)?;
            Ok(result)
        })();
        match run {
            Err(e) if p.stream && !matches!(e, GenerateError::Io) => {
                let msg = e.to_string();
                push(out, &state.stepper.fail(&msg))?;
                Err(GenerateError::Streamed(msg))
            }
            other => other,
        }
    }

    fn generate_rolling(
        &mut self,
        source: &mut dyn ContSource,
        _hold: &mut dyn FnMut(i32, Option<(u64, i32)>) -> Option<i32>,
        _store: Option<&mut KvStore>,
    ) -> Option<Vec<ContOwnedResult>> {
        let mut live = Vec::<Live<'m>>::new();
        let mut pending: Option<(ContOwnedWork, Prepared)> = None;
        let mut results = Vec::new();
        loop {
            if self.batch.poisoned() || self.stop_requested.is_some_and(|f| f()) {
                for job in live.drain(..) {
                    let item = self.abort(
                        job,
                        engine_error("V4.1 batch failed or server shutting down"),
                    );
                    Self::settle(source, &mut results, item.key, item.result);
                }
                if let Some((work, _)) = pending.take() {
                    Self::settle(
                        source,
                        &mut results,
                        work.key,
                        Err(engine_error("V4.1 batch unavailable")),
                    );
                }
                break;
            }
            // One prefill scratch allocation at a time. Poll while decode rows
            // are live, rather than waiting for their entire responses to end.
            if pending.is_none()
                && live.len() < self.batch.width()
                && live.iter().all(|j| j.state.native.ready())
            {
                if let Some(mut work) = source.next(self) {
                    match self.prepare(&work.parsed, work.prepared.as_ref(), work.default_tokens) {
                        Ok(p) if p.budget == 0 => {
                            let result = self.zero(p, &mut work);
                            Self::settle(source, &mut results, work.key, result);
                        }
                        Ok(p) => pending = Some((work, p)),
                        Err(e) => Self::settle(source, &mut results, work.key, Err(e)),
                    }
                }
            }
            if let Some((mut work, p)) = pending.take() {
                match self.batch.can_admit(&p.tokens,p.budget,p.images.as_ref()) {
                    Ok(true)=> match self.start(p,&work.parsed,&work.job_id,work.created,work.cors,work.default_tokens,work.t_arrive,work.out.as_mut()) {
                        Ok(state)=>live.push(Live { work,state }),
                        Err(e)=>Self::settle(source,&mut results,work.key,Err(e)),
                    },
                    Ok(false) if !live.is_empty()=>pending=Some((work,p)),
                    Ok(false)=>Self::settle(source,&mut results,work.key,Err(GenerateError::Capacity("V4.1 request exceeds current memory headroom; reduce prompt/output or retry"))),
                    Err(e)=>Self::settle(source,&mut results,work.key,Err(engine_error(e))),
                }
            }
            if live.is_empty() {
                break;
            }
            // Detect disconnects even while requests are in prefill or emit
            // only hidden reasoning. No busy-waiting in an empty cohort.
            let mut i = 0;
            while i < live.len() {
                let job = &mut live[i];
                let bytes = job.state.stepper.heartbeat(Instant::now());
                if let Err(e) = push(job.work.out.as_mut(), &bytes) {
                    let item = self.abort(live.remove(i), e);
                    Self::settle(source, &mut results, item.key, item.result);
                } else {
                    i += 1;
                }
            }
            // Prefill one chunk, then advance every ready row by one token.
            // A long image/text admission must not monopolize the whole decode.
            if let Some(i) = live.iter().position(|j| !j.state.native.ready()) {
                let job = &mut live[i];
                let step = match job.state.native.prefill() {
                    Err(e) => {
                        self.batch.invalidate();
                        Err(engine_error(e))
                    }
                    Ok(false) => Ok(()),
                    Ok(true) => {
                        job.state.prefill_ms = job.state.t_admit.elapsed().as_secs_f64() * 1000.;
                        self.sample_emit(&mut job.state, job.work.out.as_mut())
                    }
                };
                // Transport/sampling errors cancel this row only. A native
                // failure above poisons the GPU epoch and retires its cohort.
                if let Err(e) = step {
                    let item = self.abort(live.remove(i), e);
                    Self::settle(source, &mut results, item.key, item.result);
                }
            }
            let mut rows: Vec<_> = live
                .iter_mut()
                .filter_map(|j| j.state.pending.take().map(|t| (&mut j.state.native, t)))
                .collect();
            let n = rows.len();
            if n > 0 {
                let result = self.batch.decode(&mut rows);
                drop(rows);
                if let Err(e) = result {
                    let msg = e.to_string();
                    for job in live.drain(..) {
                        let item = self.abort(job, engine_error(&msg));
                        Self::settle(source, &mut results, item.key, item.result);
                    }
                    continue;
                }
                self.report_step(n);
                let mut i = 0;
                while i < live.len() {
                    let job = &mut live[i];
                    if job.state.native.ready() && !job.state.done {
                        job.state.steps += 1;
                        if let Err(e) = self.sample_emit(&mut job.state, job.work.out.as_mut()) {
                            let item = self.abort(live.remove(i), e);
                            Self::settle(source, &mut results, item.key, item.result);
                            continue;
                        }
                    }
                    i += 1;
                }
            }
            let mut i = 0;
            while i < live.len() {
                if !live[i].state.done {
                    i += 1;
                    continue;
                }
                let mut job = live.remove(i);
                let (tail, outcome) = self.finish(&mut job.state, job.work.cors);
                source.publish(job.work.key, &outcome);
                let result = push(job.work.out.as_mut(), &tail).map(|()| outcome);
                let key = job.work.key;
                drop(job); // free KV before the next request's admission quote
                Self::settle(source, &mut results, key, result);
            }
        }
        Some(results)
    }
}
