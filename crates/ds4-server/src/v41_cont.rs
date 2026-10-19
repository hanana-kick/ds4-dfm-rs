//! DeepSeek V4.1 native multi-request scheduler, DFM Rust HTTP host.
//!
//! YoungAi is the batch-kernel reference, not the HTTP server. Keep the
//! original DFM V4.1 serial route for roll-ineligible requests.

#![cfg(feature = "native")]

use std::ffi::c_void;
use std::io::Write;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::time::Instant;

use ds4_core::v41_media::{V41Images, V41VisionMeta};
use ds4_core::{Model, V41Batch, V41BatchRequest, V41Sampling, Vocab};
use ds4_kv::Store as KvStore;

use crate::dsml::SampleOverride;
use crate::generate::{prepare_required_prefixes, GenerateError, GenerateOutcome};
use crate::parse::{ChatPart, ParsedRequest};
use crate::render::syntax_for_model_id;
use crate::serve_cont::{
    ContExec, ContOwnedResult, ContOwnedWork, ContProbe, ContSource, ContStepper,
};
use crate::stream::ReqTimings;

/// The native context is request-bounded; this owner has one shared dense
/// Forward slab and retains individual compressed KV for every admitted job.
pub struct V41ContLane<'m> {
    model: &'m Model,
    vocab: &'m Vocab,
    batch: V41Batch<'m>,
    model_path: PathBuf,
    engram_dir: Option<String>,
    ctx: i32,
    max_rows: usize,
    vision: Option<V41VisionMeta>,
    stop: Option<fn() -> bool>,
}

struct V41Probe<'a> {
    vocab: &'a Vocab,
    template: Option<&'a ds4_core::chat_template::Template>,
    model_id: i32,
    ctx: i32,
}

impl ContProbe for V41Probe<'_> {
    fn prompt_tokens(
        &self,
        parsed: &mut ParsedRequest,
    ) -> Result<(Vec<u8>, Vec<i32>), GenerateError> {
        prepare_required_prefixes(parsed, syntax_for_model_id(self.model_id), |literal| {
            Ok(self.vocab.encode_rendered_bytes(literal))
        })?;
        let prompt = crate::chat_input::render_model(
            self.template,
            self.model_id,
            parsed,
            self.vocab.chat_system_token(),
        )?;
        let tokens = match parsed.kind {
            crate::route::ReqKind::Completion => self
                .vocab
                .encode_text(std::str::from_utf8(&prompt).unwrap_or("")),
            crate::route::ReqKind::Chat => self.vocab.encode_rendered_bytes(&prompt),
        };
        Ok((prompt, tokens))
    }

    fn seq_cap(&self) -> i32 {
        self.ctx
    }

    fn bank_live(&self, _bank: i32) -> Option<(u64, i32)> {
        // Persistent prefix banks are not qualified on V4.1.
        None
    }
}

/* Native sampling callbacks run on the owner thread while request handles are
 * live. The Box<ContStepper> pointee remains stationary as slots move.
 * The owner closes native requests before dropping either pointer. */
struct GreedyContext {
    stepper: *const ContStepper,
    vocab: *const Vocab,
}

unsafe extern "C" fn v41_greedy_cb(ud: *mut c_void, ahead: *const i32, n: u32) -> i32 {
    if ud.is_null() || (n > 0 && ahead.is_null()) || n > 16 {
        return 0;
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        let ctx = unsafe { &*(ud as *const GreedyContext) };
        let stepper = unsafe { &*ctx.stepper };
        let vocab = unsafe { &*ctx.vocab };
        let tokens: &[i32] = if n == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(ahead, n as usize) }
        };
        stepper.syntax_greedy_tokens(vocab, tokens)
    }));
    i32::from(result.unwrap_or(false))
}

struct Slot<'m> {
    work: ContOwnedWork,
    stepper: Box<ContStepper>,
    greedy_ctx: Box<GreedyContext>,
    request: Option<V41BatchRequest<'m>>,
    prefilling: bool,
    done: bool,
    engine_eos: bool,
    io_failed: bool,
    error: Option<GenerateError>,
    t_admit: Instant,
    t_first: Option<Instant>,
    decoded_steps: i32,
    prompt_n: i32,
}

impl Slot<'_> {
    fn push(&mut self, data: &[u8]) {
        if !self.io_failed && !data.is_empty() {
            if self.work.out.write_all(data).is_err() || self.work.out.flush().is_err() {
                self.io_failed = true;
                self.done = true;
            }
        }
    }

    fn on_token(&mut self, vocab: &Vocab, predicted: i32) {
        if self.done {
            return;
        }
        let token = match self.stepper.sample_override() {
            SampleOverride::Token(forced) => {
                if let Some(request) = self.request.as_mut() {
                    if let Err(e) = request.force_token(forced) {
                        self.error = Some(GenerateError::Engine(e.to_string()));
                        self.done = true;
                        return;
                    }
                }
                forced
            }
            _ => predicted, // Greedy syntax is selected in the native hook.
        };
        if self.t_first.is_none() {
            self.t_first = Some(Instant::now());
        }
        if vocab.is_stop(token) {
            self.engine_eos = true;
            self.stepper.mark_stop();
            self.done = true;
            return;
        }
        let event = self.stepper.feed(&vocab.token_text(token));
        self.push(&event.bytes);
        if event.done {
            self.done = true;
        }
    }

    fn finish(mut self) -> Result<GenerateOutcome, GenerateError> {
        // GPU and Engram buffers must be closed before the client result is
        // published; a failure never leaves an active request with stale KV.
        drop(self.request.take());
        if self.io_failed {
            return Err(GenerateError::Io);
        }
        if let Some(error) = self.error {
            return Err(error);
        }
        let mut timings = ReqTimings {
            prefill_tokens: self.prompt_n,
            decode_tokens: self.stepper.completion(),
            decode_steps: self.decoded_steps,
            ..ReqTimings::default()
        };
        if let Some(first) = self.t_first {
            timings.valid = true;
            timings.ttft_ms = first.duration_since(self.work.t_arrive).as_secs_f64() * 1e3;
            timings.prefill_ms = first.duration_since(self.t_admit).as_secs_f64() * 1e3;
        }
        let (tail, outcome) =
            self.stepper
                .finalize(self.engine_eos, 0, self.prompt_n, timings, self.work.cors);
        self.work
            .out
            .write_all(&tail)
            .map_err(|_| GenerateError::Io)?;
        self.work.out.flush().map_err(|_| GenerateError::Io)?;
        Ok(outcome)
    }
}

impl<'m> V41ContLane<'m> {
    pub fn new(
        model: &'m Model,
        model_path: PathBuf,
        engram_dir: Option<String>,
        ctx: i32,
        max_rows: usize,
        vision: Option<V41VisionMeta>,
    ) -> Result<Self, GenerateError> {
        let batch =
            V41Batch::open(model, max_rows).map_err(|e| GenerateError::Engine(e.to_string()))?;
        Ok(Self {
            model,
            vocab: model.vocab(),
            batch,
            model_path,
            engram_dir,
            ctx,
            max_rows,
            vision,
            stop: None,
        })
    }

    fn admit(&self, work: ContOwnedWork) -> Result<Slot<'m>, (ContOwnedWork, GenerateError)> {
        let prompt = match &work.prepared {
            Some(p) => (p.prompt.clone(), p.tokens.clone()),
            None => {
                return Err((
                    work,
                    GenerateError::Unsupported("missing prepared V4.1 prompt"),
                ))
            }
        };
        let mut images = None;
        if !work.parsed.images.is_empty() {
            let Some(meta) = self.vision else {
                return Err((
                    work,
                    GenerateError::BadRequest("V4.1 image input requires --vision-gguf"),
                ));
            };
            let mut ordered = Vec::new();
            for msg in &work.parsed.messages {
                for part in &msg.parts {
                    if let ChatPart::Image(index) = part {
                        ordered.push(work.parsed.images[*index].data.as_ref());
                    }
                }
            }
            match V41Images::expand(&prompt.1, &ordered, &meta, self.ctx as usize) {
                Ok(value) => images = Some(value),
                Err(_) => {
                    return Err((
                        work,
                        GenerateError::BadRequest("invalid V4.1 image spans or context length"),
                    ));
                }
            }
        }
        let tokens = images
            .as_ref()
            .map_or(prompt.1.clone(), |i| i.tokens.clone());
        let count = tokens.len() as i32;
        if count >= self.ctx || count <= 0 {
            return Err((
                work,
                GenerateError::BadRequest("V4.1 prompt exceeds context"),
            ));
        }
        let sampling = V41Sampling {
            temperature: work.parsed.temperature,
            top_k: work.parsed.top_k,
            top_p: work.parsed.top_p,
            min_p: work.parsed.min_p,
            seed: work.parsed.seed,
            freq_penalty: work.parsed.frequency_penalty,
            presence_penalty: work.parsed.presence_penalty,
            dry_multiplier: if work.parsed.dry_set {
                work.parsed.dry_multiplier
            } else {
                0.0
            },
            dry_base: if work.parsed.dry_set {
                work.parsed.dry_base
            } else {
                1.75
            },
            dry_allowed_length: if work.parsed.dry_set {
                work.parsed.dry_allowed_length
            } else {
                2
            },
        };
        let (stepper, head) = ContStepper::new(
            &work.parsed,
            self.model.model_id(),
            &work.job_id,
            work.created,
            work.cors,
            work.default_tokens,
            prompt.0,
            count,
            self.ctx,
        );
        let t_admit = Instant::now();
        let stepper = Box::new(stepper);
        let mut greedy_ctx = Box::new(GreedyContext {
            stepper: &*stepper,
            vocab: self.vocab,
        });
        let native = match V41BatchRequest::open(
            self.model,
            &self.model_path,
            self.engram_dir.as_deref(),
            tokens,
            stepper.max_tokens.max(0),
            sampling,
            images,
            Some(v41_greedy_cb),
            (&mut *greedy_ctx as *mut GreedyContext).cast::<c_void>(),
        ) {
            Ok(native) => native,
            Err(error) => return Err((work, GenerateError::Engine(error.to_string()))),
        };
        let mut slot = Slot {
            work,
            stepper,
            greedy_ctx,
            request: Some(native),
            prefilling: true,
            done: false,
            engine_eos: false,
            io_failed: false,
            error: None,
            t_admit,
            t_first: None,
            decoded_steps: 0,
            prompt_n: count,
        };
        slot.push(&head);
        Ok(slot)
    }

    fn roll(&mut self, source: &mut dyn ContSource) -> Vec<ContOwnedResult> {
        let probe = V41Probe {
            vocab: self.vocab,
            template: self.model.chat_template(),
            model_id: self.model.model_id(),
            ctx: self.ctx,
        };
        let mut slots: Vec<Slot<'m>> = Vec::new();
        let mut results = Vec::new();
        loop {
            while slots.len() < self.max_rows {
                // Keep at most one full 2048-row prefill workspace alive.
                // Other resident requests have already shrunk to KV-only.
                if slots.iter().any(|s| s.prefilling) {
                    break;
                }
                let Some(work) = source.next(&probe) else {
                    break;
                };
                match self.admit(work) {
                    Ok(slot) => slots.push(slot),
                    Err((mut work, error)) => {
                        let key = work.key;
                        // The response is still uncommitted. The owner emits
                        // the protocol-shaped error after this result.
                        let _ = work.out.flush();
                        let result = Err(error);
                        source.settled(key, &result);
                        results.push(ContOwnedResult { key, result });
                    }
                }
            }
            if slots.is_empty() {
                break;
            }
            if self.stop.is_some_and(|stop| stop()) {
                for slot in &mut slots {
                    slot.done = true;
                    slot.error = Some(GenerateError::Engine("V4.1 owner stopped".into()));
                }
            }
            // One chunk of the oldest outstanding prefill, alternating with
            // decode-ready requests. Vision spans are prepared and encoded
            // inside that native chunk; the GPU is never owned by two threads.
            if let Some(slot) = slots.iter_mut().find(|s| !s.done && s.prefilling) {
                match slot.request.as_mut().unwrap().prefill_chunk() {
                    Ok(false) => {}
                    Ok(true) => {
                        slot.prefilling = false;
                        match slot.request.as_ref().unwrap().next_token() {
                            Ok(token) => slot.on_token(self.vocab, token),
                            Err(e) => {
                                slot.error = Some(GenerateError::Engine(e.to_string()));
                                slot.done = true;
                            }
                        }
                    }
                    Err(error) => {
                        slot.error = Some(GenerateError::Engine(error.to_string()));
                        slot.done = true;
                    }
                }
            }
            // Shared GPU Forward: all ready requests contribute one row,
            // regardless of whether their original prompts contained images.
            let mut active = Vec::new();
            let mut references = Vec::new();
            for (index, slot) in slots.iter_mut().enumerate() {
                if !slot.done && !slot.prefilling {
                    active.push(index);
                    references.push(slot.request.as_mut().unwrap());
                }
            }
            if !references.is_empty() {
                let step = self.batch.step(&mut references);
                drop(references);
                match step {
                    Ok(tokens) => {
                        for (index, token) in active.into_iter().zip(tokens) {
                            let slot = &mut slots[index];
                            slot.decoded_steps += 1;
                            slot.on_token(self.vocab, token);
                        }
                    }
                    Err(error) => {
                        for index in active {
                            let slot = &mut slots[index];
                            slot.error = Some(GenerateError::Engine(error.to_string()));
                            slot.done = true;
                        }
                    }
                }
            }
            let mut i = 0;
            while i < slots.len() {
                if slots[i].done {
                    let slot = slots.remove(i);
                    let key = slot.work.key;
                    let result = slot.finish();
                    if let Ok(outcome) = &result {
                        source.publish(key, outcome);
                    }
                    source.settled(key, &result);
                    results.push(ContOwnedResult { key, result });
                } else {
                    i += 1;
                }
            }
        }
        results
    }
}

impl ContExec for V41ContLane<'_> {
    fn model_id(&self) -> i32 {
        self.model.model_id()
    }

    fn render_request(&self, parsed: &ParsedRequest) -> Result<Vec<u8>, GenerateError> {
        crate::chat_input::render_model(
            self.model.chat_template(),
            self.model.model_id(),
            parsed,
            self.vocab.chat_system_token(),
        )
    }

    fn encode_chat(&self, rendered: &[u8]) -> Vec<i32> {
        self.vocab.encode_rendered_bytes(rendered)
    }

    fn encode_text(&self, text: &str) -> Vec<i32> {
        self.vocab.encode_text(text)
    }

    // The native V4.1 request expands and validates image positions while
    // admitting; the generic Qwen-only continuous image hook must not run.
    fn prepare_tokens(
        &self,
        _parsed: &ParsedRequest,
        _prompt: &[u8],
        tokens: Vec<i32>,
    ) -> Result<Vec<i32>, GenerateError> {
        Ok(tokens)
    }

    fn seq_cap(&self) -> i32 {
        self.ctx
    }

    fn max_seq(&self) -> i32 {
        self.max_rows as i32
    }

    fn set_stop_requested(&mut self, stop_requested: Option<fn() -> bool>) {
        self.stop = stop_requested;
    }

    fn generate(
        &mut self,
        _parsed: &ParsedRequest,
        _job_id: &str,
        _created: i64,
        _cors: bool,
        _default_tokens: i32,
        _t_arrive: Instant,
        _bank_hold_retry: &mut dyn FnMut(i32, Option<(u64, i32)>) -> Option<i32>,
        _store: Option<&mut KvStore>,
        _out: &mut dyn Write,
    ) -> Result<GenerateOutcome, GenerateError> {
        Err(GenerateError::Unsupported(
            "V4.1 batch requires the Rust rolling request owner",
        ))
    }

    fn generate_rolling(
        &mut self,
        source: &mut dyn ContSource,
        _bank_hold_retry: &mut dyn FnMut(i32, Option<(u64, i32)>) -> Option<i32>,
        _store: Option<&mut KvStore>,
    ) -> Option<Vec<ContOwnedResult>> {
        Some(self.roll(source))
    }
}
