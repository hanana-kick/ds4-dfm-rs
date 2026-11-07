import fs from 'node:fs';
const fence='`'.repeat(3);
function edit(p,f){fs.writeFileSync(p,f(fs.readFileSync(p,'utf8')));}
function once(s,a,b){const n=s.split(a).length-1;if(n!==1)throw Error(`anchor count ${n}: ${a.slice(0,100)}`);return s.replace(a,b);}
function section(s,a,b,t){const i=s.indexOf(a),j=s.indexOf(b,i);if(i<0||j<0)throw Error(`section ${a}`);return s.slice(0,i)+t+s.slice(j);}
edit('ds41_batch.h',s=>{
 s=once(s,'#include <stddef.h>','#include <stddef.h>\n#include "ds41_spec_plan.h"');
 return once(s,'#ifdef __cplusplus\n}\n#endif',`/* Configure before prefill; embedded draft state belongs to this request. */
uint64_t ds4_v41_request_mtp_bytes(void *engine);
int ds4_v41_request_mtp(struct ds4_v41_request *r, int on, int verify_k);
/* Commit exactly the accepted input prefix of a pending verification. */
int ds4_v41_request_commit(struct ds4_v41_request *r, uint32_t keep);
/* Padded host output: count x DS41_VERIFY_MAX candidate ids/logit rows. */
int ds4_v41_batch_verify(struct ds4_v41_batch *b, struct ds4_v41_request **requests,
    const int32_t *tokens, const uint32_t *remaining, int count,
    int32_t *candidates, uint32_t *rows, float *logits, size_t floats);
#ifdef __cplusplus
}
#endif`);
});
edit('native/bridge/ds4_bridge_v41_batch.inc',s=>{
 s=once(s,'ds4_v41_feed_open(m->engine, np, rows, provider, wait, provider, next, provider)',
 'ds4_v41_feed_open(m->engine, np < (int)DS41_VERIFY_MAX ? (int)DS41_VERIFY_MAX : np, rows, provider, wait, provider, next, provider)');
 return s+'\n#include "ds4_bridge_v41_mtp.inc"\n';
});
edit('native/bridge/ds4_bridge.h',s=>once(s,'int ds4_bridge_session_eval_layer_slice(',`/* Owner-thread request-local MTP and ragged verification. */
uint64_t ds4_bridge_v41_request_mtp_bytes(ds4_bridge_model *m);
int ds4_bridge_v41_request_mtp(ds4_bridge_v41_request *r, int on, int k, char *err, size_t len);
int ds4_bridge_v41_request_commit(ds4_bridge_v41_request *r, uint32_t keep, char *err, size_t len);
int ds4_bridge_v41_batch_verify(ds4_bridge_v41_batch *b, ds4_bridge_v41_request *const *requests,
    const int32_t *tokens, const uint32_t *remaining, int count, int32_t *candidates,
    uint32_t *rows, float *logits, size_t floats, char *err, size_t errlen);

int ds4_bridge_session_eval_layer_slice(`));
edit('crates/ds4-sys/src/lib.rs',s=>s+`
// Request-local speculative verification; arrays use a nine-row padded stride.
unsafe extern "C" {
    pub fn ds4_bridge_v41_request_mtp_bytes(m: *mut ds4_bridge_model) -> u64;
    pub fn ds4_bridge_v41_request_mtp(r: *mut ds4_bridge_v41_request,
        on: c_int, k: c_int, err: *mut c_char, errlen: usize) -> c_int;
    pub fn ds4_bridge_v41_request_commit(r: *mut ds4_bridge_v41_request,
        keep: u32, err: *mut c_char, errlen: usize) -> c_int;
    pub fn ds4_bridge_v41_batch_verify(b: *mut ds4_bridge_v41_batch,
        requests: *const *mut ds4_bridge_v41_request, tokens: *const i32,
        remaining: *const u32, count: c_int, candidates: *mut i32, rows: *mut u32,
        logits: *mut f32, floats: usize, err: *mut c_char, errlen: usize) -> c_int;
}
`);
edit('crates/ds4-core/src/v41_run/batch.rs',s=>{
 s=once(s,'use std::rc::Rc;','use std::rc::Rc;\nmod mtp;\nuse mtp::VERIFY_MAX;');
 s=once(s,'    floor_bytes: u64,\n    logits:','    floor_bytes: u64,\n    mtp: bool,\n    verify_k: i32,\n    logits:');
 s=once(s,'    generated: Vec<i32>,','    generated: Vec<i32>,\n    verify_ids: Vec<i32>,\n    verify_row: usize,\n    spec_rounds: u64,\n    spec_offered: u64,\n    spec_accepted: u64,');
 s=once(s,'            logits: vec![0.; width * vocab],','            mtp: false,\n            verify_k: 0,\n            logits: vec![0.; width * VERIFY_MAX * vocab],');
 s=once(s,'        let peak = peak\n            .checked_add(image_peak)',`        let extra = if self.mtp {
            // SAFETY: immutable metadata query; weights are shared.
            unsafe { ffi::ds4_bridge_v41_request_mtp_bytes(self.model.raw_ptr()) }
        } else { 0 };
        let peak = peak.checked_add(extra)
            .ok_or_else(|| asset_error("MTP allocation quote overflow"))?;
        let peak = peak
            .checked_add(image_peak)`);
 s=once(s,'        Ok(V41Request {',`        if self.mtp {
            // SAFETY: prefill has not started; release on failure while the
            // boxed provider and borrowed image patches are still alive.
            let rc = unsafe { ffi::ds4_bridge_v41_request_mtp(raw.as_ptr(), 1,
                self.verify_k, err.as_mut_ptr().cast(), err.len()) };
            if rc != 0 {
                unsafe { ffi::ds4_bridge_v41_request_close(raw.as_ptr()); }
                return Err(ffi_error(&err));
            }
        }
        Ok(V41Request {`);
 s=once(s,'            generated: Vec::new(),','            generated: Vec::new(),\n            verify_ids: Vec::new(),\n            verify_row: 0,\n            spec_rounds: 0,\n            spec_offered: 0,\n            spec_accepted: 0,');
 s=once(s,`        if self
            .logits
            .iter()`,`        let vocab = self.model.vocab().n_vocab() as usize;
        let start = self.verify_row.checked_mul(vocab)
            .ok_or_else(|| asset_error("verification row overflow"))?;
        let selected = self.logits.get(start..start + vocab)
            .ok_or_else(|| asset_error("missing verification logit row"))?;
        if selected
            .iter()`);
 s=once(s,'!self.logits.iter().any(|x| x.is_finite())','!selected.iter().any(|x| x.is_finite())');
 s=once(s,'            r.logits.copy_from_slice(logits);','            r.logits.clear();\n            r.logits.extend_from_slice(logits);\n            r.verify_row = 0;');
 s=once(s,'        let mut row = self.logits.clone();','        let mut row = selected.to_vec();');
 s=s.replace('A shared, plain-decode workspace. No persistent banks, graphs or MTP here.','A shared packed decode/verification workspace. Requests own their draft states.');
 s=s.replace('Maximum packed rows of the existing V4.1 GEMV kernel path, not a throughput\n/// or GPU qualification claim. Larger batches need a separately gated kernel.','Maximum active requests; verification row capacity is nine times this count.\n/// C4 with five draft tokens uses 24 verification rows.');
 return s;
});
edit('crates/ds4-server/src/serve_v41.rs',s=>{
 s=s.replace('//! Experimental V4.1 rolling requests with shared packed dense/MoE execution.','//! V4.1 rolling requests with shared packed dense/MoE verification.');
 s=s.replace('//! layout. This lane deliberately has no MTP or cross-request KV checkpoints.','//! layout. MTP state is request-local; cross-request KV checkpoints are separate.');
 s=once(s,'    max_rows: usize,','    max_rows: usize,\n    max_draft_rows: usize,');
 s=once(s,'    steps: i32,','    steps: i32,\n    output_budget: usize,');
 s=once(s,'if route.dspark != Some(0) || !route.no_kv_reuse {','if !route.no_kv_reuse {');
 s=once(s,'V4.1 concurrent lane requires --mtp-mode off --prefix-reuse off','V4.1 concurrent lane requires --prefix-reuse off');
 s=once(s,'        let batch = V41Batch::new(','        let mut batch = V41Batch::new(');
 s=once(s,`        .map_err(engine_error)?;
        Ok(Self {`,`        .map_err(engine_error)?;
        if route.dspark != Some(0) {
            batch.enable_mtp(route.verify_k, route.graph.unwrap_or(true)).map_err(engine_error)?;
        }
        Ok(Self {`);
 s=once(s,'            max_rows: 0,','            max_rows: 0,\n            max_draft_rows: 0,');
 s=once(s,'            steps: 0,\n        })\n    }\n\n    fn zero(','            steps: 0,\n            output_budget: prepared.budget,\n        })\n    }\n\n    fn zero(');
 s=once(s,'    fn finish(&self,',`    /// Target draws use the live DSML/EOS/penalty policy at each row.
    /// Rejected drafts leave neither KV nor Engram history behind.
    fn finish_verify_round(&mut self, state: &mut State<'m>, out: &mut dyn Write) -> Result<(), GenerateError> {
        let rows = state.native.verify_rows();
        let mut keep = 0;
        for row in 0..rows {
            state.native.select_verify_row(row).map_err(engine_error)?;
            self.sample_emit(state, out)?;
            keep = row + 1;
            if state.done || row + 1 == rows
                || state.pending != state.native.verify_candidate(row + 1) { break; }
        }
        state.native.commit_verify(keep).map_err(|e| {
            self.batch.invalidate();
            engine_error(e)
        })
    }

    fn finish(&self,`);
 s=once(s,'        let (tail, result) = state.stepper.finalize(state.eos, 0, n, t, cors);',`        let (tail, mut result) = state.stepper.finalize(state.eos, 0, n, t, cors);
        let (rounds, offered, accepted) = state.native.spec_stats();
        result.speculation_active = offered > 0;
        if (offered > 0) {
            eprintln!("[v41-batch] MTP rounds={rounds} offered={offered} accepted={accepted}");
        }`);
 s=once(s,'    fn report_step(&mut self, n: usize) {','    fn report_step(&mut self, n: usize, rows: usize) {');
 s=once(s,'        self.steps += 1;\n        if n > self.max_rows {',`        self.steps += 1;
        let drafts = rows.saturating_sub(n);
        if n > self.max_rows || drafts > self.max_draft_rows {
            eprintln!("[v41-batch] packed_verify sequences={n} rows={rows} draft_rows={drafts}");
            self.max_draft_rows = self.max_draft_rows.max(drafts);
        }
        if n > self.max_rows {`);
 s=once(s,`                    self.batch
                        .decode(&mut [(&mut state.native, token)])
                        .map_err(engine_error)?;
                    self.report_step(1);
                    state.steps += 1;`,`                    let remaining = state.output_budget.saturating_sub(state.stepper.completion().max(0) as usize);
                    let rows = self.batch.verify(&mut [(&mut state.native, token, remaining)])
                        .map_err(engine_error)?;
                    self.report_step(1, rows);
                    state.steps += 1;
                    self.finish_verify_round(&mut state, out)?;
                    continue;`);
 s=once(s,'.filter_map(|j| j.state.pending.take().map(|t| (&mut j.state.native, t)))',`.filter_map(|j| {
                    let remaining = j.state.output_budget.saturating_sub(j.state.stepper.completion().max(0) as usize);
                    j.state.pending.take().map(|t| (&mut j.state.native, t, remaining))
                })`);
 s=once(s,'                let result = self.batch.decode(&mut rows);','                let result = self.batch.verify(&mut rows);');
 s=once(s,'                self.report_step(n);','                self.report_step(n, result.expect("error handled above"));');
 s=once(s,'if job.state.native.ready() && !job.state.done {','if job.state.native.verify_rows() > 0 && !job.state.done {');
 s=once(s,'if let Err(e) = self.sample_emit(&mut job.state, job.work.out.as_mut()) {','if let Err(e) = self.finish_verify_round(&mut job.state, job.work.out.as_mut()) {');
 return s.replace('Prefill one chunk, then advance every ready row by one token.','Prefill one chunk, then verify every ready request in one packed batch.');
});
edit('crates/ds4-core/src/serving.rs',s=>{
 const at=s.indexOf('if variant == Variant::DeepSeek41Flash {');
 const end=s.indexOf('    if variant == Variant::K2Horizon375B',at);
 if(at<0||end<0)throw Error('V41 caps anchor');
 const caps=once(s.slice(at,end),'spec_lane: SpecLane::Serial','spec_lane: SpecLane::Both');
 s=s.slice(0,at)+caps+s.slice(end);
 s=once(s,`        if req.mtp_mode != MtpMode::Off {
            issues.push(error(
                "v41_batch_mtp",
                "V4.1 concurrent decoding requires explicit --mtp-mode off",
            ));
        }
`,'');
 s=once(s,'experimental V4.1 packed lane: plain eager decode, request-local images/Engram, no MTP/reuse/snapshots; GPU qualification pending','V4.1 packed verification: request-local MTP/images/Engram, eager target verification; GPU qualification pending');
 const ti=s.indexOf('    fn v41_serving_contract_refuses_what_it_lacks()');
 const te=s.indexOf('    fn v41_packed_requests_require_explicit_bounded_plain_lane()',ti);
 if(ti<0||te<0)throw Error('V41 test anchors');
 s=s.slice(0,ti)+once(s.slice(ti,te),'assert_eq!(caps.spec_lane, SpecLane::Serial);','assert_eq!(caps.spec_lane, SpecLane::Both);')+s.slice(te);
 s=once(s,'        assert!(p.issues.iter().any(|i| i.code == "v41_batch_mtp"));','        assert!(!p.issues.iter().any(|i| i.code == "v41_batch_mtp"));');
 s=once(s,'opt-in 2..8 ephemeral requests require --mtp-mode off --prefix-reuse off and have no GPU qualification yet','opt-in 2..8 ephemeral requests support embedded MTP with --prefix-reuse off and have no GPU qualification yet');
 s=once(s,'.is_some_and(|caps| caps.spec_lane == SpecLane::Serial)', '.is_some_and(|caps| caps.spec_lane == SpecLane::Serial || (caps.variant == Variant::DeepSeek41Flash && self.effective.max_seqs == 1))');
 s=once(s,'        mtp: if mtp_weights {\n            caps.mtp_support','        mtp: if mtp_weights && caps.variant == Variant::DeepSeek41Flash && max_seqs > 1 {\n            Support::Present\n        } else if mtp_weights {\n            caps.mtp_support');
 const test=`    #[test]
    fn v41_concurrent_mtp_is_not_downgraded_to_serial() {
        for width in [2, 4, 8] {
            for mode in [MtpMode::On, MtpMode::Auto] {
                let p = plan(ServingRequest {
                    max_seqs: MaxSeqs::Fixed(width), mtp_mode: mode,
                    prefix_reuse: PrefixReuse::Off, ..ServingRequest::default()
                }, ModelFamily::DeepSeek4, Variant::DeepSeek41Flash);
                assert!(!p.has_errors(), "{width}: {:?}", p.issues);
                assert_eq!(p.effective.max_seqs, width);
                assert_eq!(p.effective.mtp_mode, mode);
                assert!(p.effective.mtp_weights);
                assert!(!p.uses_serial_mtp());
                assert!(p.wants_bank_lane());
                assert_eq!(p.qualified.mtp, Support::Present);
            }
        }
    }

`;
 return once(s,'    #[test]\n    fn v41_draft_scheduler_no_width()',test+'    #[test]\n    fn v41_draft_scheduler_no_width()');
});
edit('crates/ds4-server/src/bin/ds4-server-rs.rs',s=>once(s,
`            eprintln!("[v41-batch] experimental lane width={} vision={} MTP=off reuse=off; GPU qualification pending",
                cont_width, v41_route.as_ref().is_some_and(|r| r.vision.is_some()));`,
`            eprintln!("[v41-batch] lane width={} vision={} MTP={} reuse=off",
                cont_width, v41_route.as_ref().is_some_and(|r| r.vision.is_some()),
                fitted.effective.mtp_mode.as_str());`));
edit('tests/parity/bridge_null_stubs.c',s=>s+`
uint64_t ds4_v41_request_mtp_bytes(void *e) { (void)e;return UINT64_MAX; }
int ds4_v41_request_mtp(struct ds4_v41_request *r,int on,int k) { (void)r;(void)on;(void)k;return 1; }
int ds4_v41_request_commit(struct ds4_v41_request *r,uint32_t n) { (void)r;(void)n;return 1; }
int ds4_v41_batch_verify(struct ds4_v41_batch *b,struct ds4_v41_request **r,const int32_t *t,
    const uint32_t *rem,int n,int32_t *ids,uint32_t *nr,float *l,size_t f) {
    (void)b;(void)r;(void)t;(void)rem;(void)n;(void)ids;(void)nr;(void)l;(void)f;return 1;
}
`);
edit('Makefile',s=>s+`
.PHONY: test-ds41-spec-plan
test-ds41-spec-plan:
	$(CC) $(CFLAGS) -I. tests/test_ds41_spec_plan.c -o /tmp/ds41-spec-plan
	/tmp/ds41-spec-plan

gate-ds41: test-ds41-spec-plan
`);
edit('.github/workflows/ds41-port-checks.yml',s=>once(s,'make test-ds41-bridge-lifetime test-ds41-batch-lifetime','make test-ds41-bridge-lifetime test-ds41-batch-lifetime test-ds41-spec-plan'));
edit('tests/ds41_concurrent_http_gate.sh',s=>{
 s=once(s,'--mem-floor-gb 2 --mtp-mode off --prefix-reuse off','--mem-floor-gb 2 --prefix-reuse off');
 s=once(s,'"${common[@]}" --max-seqs "$width"','"${common[@]}" --max-seqs "$width" --mtp-mode "$([[ "$width" == 1 ]] && echo off || echo on)"');
 const evidence=`
# C4 configuration alone is not evidence of executed speculative verification.
awk '/\\[v41-batch\\] packed_verify sequences=4 / { for(i=1;i<=NF;i++) if($i ~ /^draft_rows=/) { split($i,a,"="); if(a[2]>0) found=1 } } END { exit !found }' "$OUT/server-c4.log" || fail 'no actual C4 MTP verification observed'
`;
 return once(s,`kill -0 "$PID" || fail 'server died'`,`kill -0 "$PID" || fail 'server died'`+evidence);
});
edit('docs/ds41-concurrent-serving.md',s=>{
 s=s.replace('# Experimental V4.1 concurrent serving (fork-only)','# V4.1 concurrent serving and embedded MTP');
 s=s.replace('The plain packed-row formulation is adapted to this tree\'s existing','The packed-row formulation is adapted to this tree\'s existing');
 s=s.replace('global sampler state, speculative batching\nand multiple CUDA attention streams are deliberately not imported.','global sampler state and multiple CUDA attention streams are not imported.\nThe target sampler stays in Rust; each request owns its embedded drafter,\nverification frontier and rollback state.');
 s=s.replace('A decode step packs one token from each ready request into','A decode step packs the confirmed token and up to five draft tokens from each\nready request into');
 s=s.replace('| MTP | existing embedded DSpark | **must set `--mtp-mode off`** |','| MTP | existing embedded DSpark | **on/auto/off**, request-local drafts and packed target verification |');
 s=s.replace('| CUDA graphs | existing serial path | eager packed decode only |','| CUDA graphs | existing serial path | eager packed target verification; request-local draft graphs |');
 s=section(s,'**Engine limitation:**','## Validation',`C4 + MTP is implemented: four five-token draft blocks produce up to 24 target
verification rows in one dense/MoE batch. Request count and row capacity are
separate. The same path accepts up to eight active requests. Near context or
output limits, each request independently shortens its verification block.
The launcher does not turn MTP off or fall back to C1.

Target sampling is performed row by row in Rust, with each request's RNG,
penalties, forced DSML syntax and EOS/stop policy. Only a matching draft prefix
is accepted. A mismatch emits the target's own token and restores rejected KV,
SWA/compressor state, main-hidden history and Engram history before the next
round. Images stay request-local. A late image tower pass is not preemptive.

Execution logs distinguish configuration from work:
${fence}text
[v41-batch] packed_verify sequences=4 rows=24 draft_rows=20
[v41-batch] MTP rounds=... offered=... accepted=...
${fence}

Model files are unchanged. Run \`scripts/build.sh\` after pulling, then
\`scripts/serve.sh\`. Prefix/disk KV reuse remains separate and off for this
request-local lane. CUDA/model execution requires a real GPU run; host checks
are not evidence that reasoning repetition has been solved.

`);
 return s.replace('compares serial\nplain-decode against C4,','compares serial\nplain-decode against C4 with MTP enabled,');
});
console.log('Applied request-local DSpark, packed verification, rollback and sampling.');
