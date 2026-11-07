//! Owner-thread-affine V4.1 request states, sharing weights and packed decode.
//!
//! Unlike the serial asset cache, each active request opens an independent
//! FeedCore (history, image mask, io_uring task and pinned destinations). The
//! serial cache's Arc<Mutex<FeedCore>> MUST NOT be reused by concurrent requests.
//! Native code does prefill/packed forward; Rust owns sampling and admission.
use super::*;
use crate::sample::{draw, Sampler};
use crate::v41_media::V41Images;
use ds4_sys as ffi;
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::rc::Rc;

mod round;
use round::{VerifiedRound, VERIFY_MAX};

/// Maximum active requests. Each may contribute 1 + the trained MTP block
/// target rows; native projection tiles preserve the decode arithmetic path.
pub const V41_BATCH_MAX: usize = 8;

/// Shared target verification workspace, with request-local MTP state.
/// Requests borrow the same model and cannot be moved to another host thread.
pub struct V41Batch<'m> {
    raw: NonNull<ffi::ds4_bridge_v41_batch>,
    model: &'m Model,
    model_path: std::path::PathBuf,
    engram_dir: Option<String>,
    no_engram: bool,
    width: usize,
    floor_bytes: u64,
    logits: Vec<f32>,
    mtp_mode: i32,
    verify_k: u32,
    graph: bool,
    last_rows: usize,
    poisoned: bool,
    _thread: PhantomData<Rc<()>>,
}

/// One isolated KV/Engram/image/sampling history. All foreign borrows point
/// into stable Box/Vec allocations owned here, not into a movable stack frame.
pub struct V41Request<'m> {
    raw: NonNull<ffi::ds4_bridge_v41_request>,
    model: &'m Model,
    // Drop below destroys native/feed first, before these borrowed bytes.
    _feed: Option<Box<V41Feed>>,
    images: Option<V41Images>,
    logits: Vec<f32>,
    verify_logits: Vec<f32>,
    round: VerifiedRound,
    spec_rounds: u64,
    spec_offered: u64,
    spec_accepted: u64,
    generated: Vec<i32>,
    breakers: Option<Vec<bool>>,
    ready: bool,
    poisoned: bool,
    _thread: PhantomData<Rc<()>>,
}

fn ffi_error(buf: &[u8]) -> Error {
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    asset_error(String::from_utf8_lossy(&buf[..end]))
}

/// Pinned callback userdata is the boxed feed, NOT the serial FeedTramp that
/// borrows a stack-local feed. Every callback catches unwind at the FFI edge.
unsafe extern "C" fn batch_rows(
    pos: u32,
    tokens: *const i32,
    n: i32,
    dst: *const *mut u8,
    ud: *mut c_void,
) -> i32 {
    if ud.is_null() || tokens.is_null() || dst.is_null() || n < 1 {
        return 1;
    }
    // SAFETY: bridge retains the Box until close; sizes are the native feed's
    // checked block and the model's Engram layer count.
    let feed = unsafe { &mut *ud.cast::<V41Feed>() };
    let ids = unsafe { std::slice::from_raw_parts(tokens, n as usize) };
    let ptrs = unsafe { std::slice::from_raw_parts(dst, feed.n_layers) };
    match catch_unwind(AssertUnwindSafe(|| feed.prepare(pos, ids, ptrs))) {
        Ok(Ok(())) => 0,
        _ => 1,
    }
}
unsafe extern "C" fn batch_wait(k: i32, ud: *mut c_void) -> i32 {
    if ud.is_null() || k < 0 {
        return 1;
    }
    // SAFETY: same stable provider and owner thread as batch_rows.
    let feed = unsafe { &mut *ud.cast::<V41Feed>() };
    match catch_unwind(AssertUnwindSafe(|| feed.wait_layer(k as u32))) {
        Ok(Ok(())) => 0,
        _ => 1,
    }
}
unsafe extern "C" fn batch_next(pos: u32, tokens: *const i32, n: i32, ud: *mut c_void) -> i32 {
    if ud.is_null() || tokens.is_null() || n < 0 {
        return 1;
    }
    if n == 0 {
        return 0;
    }
    // SAFETY: checked native slice, callback does not retain its token pointer.
    let feed = unsafe { &mut *ud.cast::<V41Feed>() };
    let ids = unsafe { std::slice::from_raw_parts(tokens, n as usize) };
    match catch_unwind(AssertUnwindSafe(|| feed.prefetch_next(pos, ids))) {
        Ok(Ok(())) => 0,
        _ => 1,
    }
}
unsafe extern "C" fn batch_drain(ud: *mut c_void) -> i32 {
    if ud.is_null() {
        return 1;
    }
    // SAFETY: drain runs before native pinned buffers and this Box are freed.
    let feed = unsafe { &mut *ud.cast::<V41Feed>() };
    match catch_unwind(AssertUnwindSafe(|| feed.drain())) {
        Ok(Ok(())) => 0,
        Ok(Err(_)) => 1,
        // A panic cannot establish that asynchronous writers have stopped.
        Err(_) => std::process::abort(),
    }
}

fn image_view(
    images: Option<&V41Images>,
) -> (Vec<ffi::ds4_bridge_v41_image>, ffi::ds4_bridge_v41_images) {
    let raw: Vec<_> = images.map_or_else(Vec::new, |imgs| {
        imgs.images
            .iter()
            .map(|im| ffi::ds4_bridge_v41_image {
                patches: im.patches.as_ptr(),
                patches_len: im.patches.len() as u64,
                n_vit_h: im.plan.n_vit_h,
                n_vit_w: im.plan.n_vit_w,
                n_llm_h: im.plan.n_llm_h,
                n_llm_w: im.plan.n_llm_w,
            })
            .collect()
    });
    let view = ffi::ds4_bridge_v41_images {
        img: raw.as_ptr(),
        n: raw.len() as i32,
        row_of: images.map_or(std::ptr::null(), |i| i.row_of.as_ptr()),
        sig_of: images.map_or(std::ptr::null(), |i| i.sig_of.as_ptr()),
        np: images.map_or(0, |i| i.np),
        n_rows: images.map_or(0, |i| i.n_rows),
    };
    (raw, view)
}

fn admission_fits(peak: u64, available: u64, floor: u64) -> bool {
    peak.checked_mul(2)
        .and_then(|p| p.checked_add(floor))
        .is_some_and(|need| need <= available)
}

impl<'m> V41Batch<'m> {
    /// Mount immutable GR/RB once. Caller must not change model/sidecars while
    /// requests are live. Serial/default serving is not changed by this lane.
    pub fn new(
        model: &'m Model,
        path: &Path,
        engram_dir: Option<&str>,
        no_engram: bool,
        zchain: Option<&str>,
        posttrain: Option<&str>,
        width: usize,
        floor_bytes: u64,
    ) -> Result<Self> {
        if model.model_id() != crate::Variant::DeepSeek41Flash as i32
            || !(1..=V41_BATCH_MAX).contains(&width)
        {
            return Err(asset_error(
                "V4.1 packed lane requires a V4.1 model and width 1..8",
            ));
        }
        v41_mount_zchain(model, zchain, posttrain)?;
        let vocab = model.vocab().n_vocab() as usize;
        let mut err = [0u8; 512];
        // SAFETY: opened model outlives the returned handle; native validates width.
        let raw = unsafe {
            ffi::ds4_bridge_v41_batch_open(
                model.raw_ptr(),
                width as i32,
                err.as_mut_ptr().cast(),
                err.len(),
            )
        };
        let raw = NonNull::new(raw).ok_or_else(|| ffi_error(&err))?;
        Ok(Self {
            raw,
            model,
            model_path: path.into(),
            engram_dir: engram_dir.map(str::to_owned),
            no_engram,
            width,
            floor_bytes,
            logits: vec![0.; width * VERIFY_MAX * vocab],
            mtp_mode: 0,
            verify_k: 0,
            graph: true,
            last_rows: 0,
            poisoned: false,
            _thread: PhantomData,
        })
    }
    /// Select before admitting requests. On requires a real per-request
    /// drafter; a missing/failed allocation never silently becomes C1/plain.
    pub fn with_mtp(mut self, mode: i32, verify: i32, graph: bool) -> Result<Self> {
        if !(0..=2).contains(&mode) || !(0..VERIFY_MAX as i32).contains(&verify) {
            return Err(asset_error("invalid V4.1 MTP mode or verification width"));
        }
        self.mtp_mode = mode;
        self.verify_k = verify as u32;
        self.graph = graph;
        Ok(self)
    }
    pub fn last_verified_rows(&self) -> usize { self.last_rows }
    pub fn width(&self) -> usize {
        self.width
    }
    pub fn poisoned(&self) -> bool {
        self.poisoned
    }
    /// An upstream GPU/prefill failure invalidates the entire owner epoch.
    pub fn invalidate(&mut self) {
        self.poisoned = true;
    }

    /// Current host headroom, including this request's images. False means
    /// wait for a live slot to retire; do not silently disable Engram/vision.
    pub fn can_admit(
        &self,
        prompt: &[i32],
        output: usize,
        images: Option<&V41Images>,
    ) -> Result<bool> {
        if self.poisoned {
            return Err(asset_error("V4.1 batch is poisoned; restart the server"));
        }
        if prompt.is_empty()
            || prompt.len() > i32::MAX as usize
            || output == 0
            || output > i32::MAX as usize
        {
            return Err(asset_error("invalid V4.1 request token budget"));
        }
        if let Some(imgs) = images {
            imgs.validate(prompt)?;
        }
        let (_raw, view) = image_view(images);
        // SAFETY: no retained pointers, immutable checked inputs for both quotes.
        let peak = unsafe {
            ffi::ds4_bridge_v41_request_peak(
                self.model.raw_ptr(),
                prompt.len() as i32,
                output as i32,
                i32::from(self.no_engram),
            )
        };
        let image_peak = if images.is_some() {
            unsafe { ffi::ds4_bridge_v41_images_bytes(&view) }
        } else {
            0
        };
        let peak = peak
            .checked_add(image_peak)
            .ok_or_else(|| asset_error("V4.1 context or allocation quote overflow"))?;
        if peak == u64::MAX {
            return Err(asset_error("V4.1 request exceeds model context"));
        }
        let avail = mem_available_mb()
            .and_then(|m| u64::try_from(m).ok())
            .ok_or_else(|| asset_error("cannot read host MemAvailable for V4.1 admission"))?;
        Ok(admission_fits(
            peak,
            avail.saturating_mul(1 << 20),
            self.floor_bytes,
        ))
    }

    pub fn request(
        &self,
        prompt: Vec<i32>,
        output: usize,
        images: Option<V41Images>,
    ) -> Result<V41Request<'m>> {
        if !self.can_admit(&prompt, output, images.as_ref())? {
            return Err(asset_error("V4.1 request cannot fit current host headroom"));
        }
        // Deliberately bypass open()/from_assets(): that serial fast path
        // shares FeedCore, and clears another active request's token history.
        let mut feed = if self.no_engram {
            None
        } else {
            V41Feed::open_uncached(&self.model_path, self.engram_dir.as_deref())?.map(Box::new)
        };
        if let Some(f) = &mut feed {
            f.seed_prefix(&prompt);
            let mut core = f
                .core
                .lock()
                .map_err(|_| asset_error("V4.1 feed lock poisoned"))?;
            core.image_mask.clear();
            if let Some(imgs) = &images {
                core.image_mask
                    .extend(imgs.sig_of.iter().map(|&s| u8::from(s != 0)));
            }
        }
        let (_raw_images, view) = image_view(images.as_ref());
        let provider = feed
            .as_deref_mut()
            .map_or(std::ptr::null_mut(), |f| (f as *mut V41Feed).cast());
        let mut err = [0u8; 512];
        // SAFETY: prompt is copied natively; image patches/boxed provider
        // remain owned by the returned request until its native destructor.
        let raw = unsafe {
            ffi::ds4_bridge_v41_request_open(
                self.model.raw_ptr(),
                prompt.as_ptr(),
                prompt.len() as i32,
                output as i32,
                i32::from(self.no_engram),
                if images.is_some() {
                    &view
                } else {
                    std::ptr::null()
                },
                feed.as_ref().map(|_| {
                    batch_rows
                        as unsafe extern "C" fn(
                            u32,
                            *const i32,
                            i32,
                            *const *mut u8,
                            *mut c_void,
                        ) -> i32
                }),
                feed.as_ref()
                    .filter(|f| f.async_protocol())
                    .map(|_| batch_wait as unsafe extern "C" fn(i32, *mut c_void) -> i32),
                feed.as_ref().filter(|f| f.async_protocol()).map(|_| {
                    batch_next as unsafe extern "C" fn(u32, *const i32, i32, *mut c_void) -> i32
                }),
                feed.as_ref()
                    .map(|_| batch_drain as unsafe extern "C" fn(*mut c_void) -> i32),
                provider,
                err.as_mut_ptr().cast(),
                err.len(),
            )
        };
        let raw = NonNull::new(raw).ok_or_else(|| ffi_error(&err))?;
        // SAFETY: configure before prefill; feed/images are still owned here.
        let rc = unsafe {
            ffi::ds4_bridge_v41_request_set_mtp(
                raw.as_ptr(), self.mtp_mode, self.verify_k, i32::from(self.graph),
                err.as_mut_ptr().cast(), err.len(),
            )
        };
        if rc != 0 {
            unsafe { ffi::ds4_bridge_v41_request_close(raw.as_ptr()); }
            return Err(ffi_error(&err));
        }
        Ok(V41Request {
            raw,
            model: self.model,
            _feed: feed,
            images,
            logits: vec![0.; self.model.vocab().n_vocab() as usize],
            verify_logits: vec![0.; VERIFY_MAX * self.model.vocab().n_vocab() as usize],
            round: VerifiedRound::default(),
            spec_rounds: 0,
            spec_offered: 0,
            spec_accepted: 0,
            generated: Vec::new(),
            breakers: None,
            ready: false,
            poisoned: false,
            _thread: PhantomData,
        })
    }

    /// One logical packed target forward. MTP adds request-local proposals;
    /// target samples/DSML decisions and rollback are committed by the host.
    pub fn decode(&mut self, rows: &mut [(&mut V41Request<'m>, i32)]) -> Result<()> {
        if self.poisoned || rows.is_empty() || rows.len() > self.width {
            return Err(asset_error("invalid/poisoned V4.1 cohort"));
        }
        for (request, token) in rows.iter() {
            if !request.ready || request.poisoned || request.round.rows() != 0
                || !std::ptr::eq(request.model, self.model)
                || *token < 0 || *token >= self.model.vocab().n_vocab()
            {
                return Err(asset_error("unready V4.1 cohort or uncommitted verification"));
            }
        }
        let handles: Vec<_> = rows.iter().map(|(r, _)| r.raw.as_ptr()).collect();
        let tokens: Vec<_> = rows.iter().map(|(_, t)| *t).collect();
        let vocab = self.model.vocab().n_vocab() as usize;
        let mut proposals = [0i32; V41_BATCH_MAX * VERIFY_MAX];
        let mut widths = [1u32; V41_BATCH_MAX];
        let mut err = [0u8; 512];
        // SAFETY: exclusive owner-thread requests; all output arrays cover
        // the maximum trained width. Native execution synchronizes on return.
        let rc = unsafe {
            if self.mtp_mode != 0 {
                ffi::ds4_bridge_v41_batch_verify(
                    self.raw.as_ptr(), handles.as_ptr(), tokens.as_ptr(), rows.len() as i32,
                    self.logits.as_mut_ptr(), rows.len() * VERIFY_MAX * vocab,
                    proposals.as_mut_ptr(), proposals.len(), widths.as_mut_ptr(),
                    err.as_mut_ptr().cast(), err.len(),
                )
            } else {
                ffi::ds4_bridge_v41_batch_step(
                    self.raw.as_ptr(), handles.as_ptr(), tokens.as_ptr(), rows.len() as i32,
                    self.logits.as_mut_ptr(), rows.len() * vocab,
                    err.as_mut_ptr().cast(), err.len(),
                )
            }
        };
        let widths_ok = widths[..rows.len()].iter().all(|&n| (1..=VERIFY_MAX as u32).contains(&n));
        if rc != 0 || !widths_ok {
            self.poisoned = true;
            for (r, _) in rows { r.poisoned = true; }
            return Err(if rc != 0 { ffi_error(&err) } else { asset_error("invalid native MTP row count") });
        }
        let mut offset = 0;
        self.last_rows = 0;
        for (i, (r, _)) in rows.iter_mut().enumerate() {
            let n = widths[i] as usize;
            r.logits.copy_from_slice(&self.logits[offset..offset + vocab]);
            if self.mtp_mode != 0 {
                if !r.round.begin(&proposals[i * VERIFY_MAX..i * VERIFY_MAX + n]) {
                    self.poisoned = true; r.poisoned = true;
                    return Err(asset_error("overlapping V4.1 verification rounds"));
                }
                r.verify_logits[..n * vocab].copy_from_slice(&self.logits[offset..offset + n * vocab]);
                if n > 1 { r.spec_rounds += 1; r.spec_offered += (n - 1) as u64; }
            }
            offset += n * vocab;
            self.last_rows += n;
        }
        Ok(())
    }
}
impl Drop for V41Batch<'_> {
    fn drop(&mut self) {
        // SAFETY: exact matching destructor on its owner thread.
        unsafe {
            ffi::ds4_bridge_v41_batch_close(self.raw.as_ptr());
        }
    }
}
impl V41Request<'_> {
    pub fn prefill(&mut self) -> Result<bool> {
        if self.ready || self.poisoned {
            return Err(asset_error("V4.1 request not in prefill phase"));
        }
        let mut err = [0u8; 512];
        // SAFETY: exclusively owned handle and full host vocabulary row.
        let rc = unsafe {
            ffi::ds4_bridge_v41_request_prefill(
                self.raw.as_ptr(),
                self.logits.as_mut_ptr(),
                self.logits.len(),
                err.as_mut_ptr().cast(),
                err.len(),
            )
        };
        if rc < 0 {
            self.poisoned = true;
            return Err(ffi_error(&err));
        }
        self.ready = rc == 1;
        if self.ready {
            self.images = None;
        }
        Ok(self.ready)
    }
    pub fn position(&self) -> usize {
        // SAFETY: live immutable request metadata, no GPU access.
        unsafe { ffi::ds4_bridge_v41_request_position(self.raw.as_ptr()) as usize }
    }
    pub fn ready(&self) -> bool {
        self.ready
    }

    /// Called only after this target token passed the real host's grammar,
    /// EOS, tool-stop and output-budget checks. Future rows are never sampled.
    pub fn advance_verified(&mut self, token: i32) -> Result<bool> {
        if self.poisoned { return Err(asset_error("poisoned V4.1 verification")); }
        if !self.round.accept(token) { return Ok(false); }
        let vocab = self.logits.len();
        let offset = self.round.cursor() * vocab;
        self.logits.copy_from_slice(&self.verify_logits[offset..offset + vocab]);
        Ok(true)
    }
    pub fn finish_round(&mut self) -> Result<()> {
        let keep = self.round.keep();
        if keep == 0 { return Ok(()); } // prefill or ordinary decode
        let mut err = [0u8; 512];
        // SAFETY: this request exclusively owns its provisional target state.
        let rc = unsafe {
            ffi::ds4_bridge_v41_request_commit(
                self.raw.as_ptr(), keep as u32, err.as_mut_ptr().cast(), err.len(),
            )
        };
        if rc != 0 { self.poisoned = true; return Err(ffi_error(&err)); }
        self.spec_accepted += (keep - 1) as u64;
        self.round.clear();
        Ok(())
    }
    pub fn spec_stats(&self) -> (u64, u64, u64) {
        (self.spec_rounds, self.spec_offered, self.spec_accepted)
    }

    /// Common Rust sampler; coins and penalties are request-local. Non-greedy
    /// streams need not equal the old CUDA sampler's seed stream. Greedy
    /// C1/C4 equality remains a required GPU gate, not a compile-time claim.
    pub fn sample(
        &mut self,
        face: V41Sampling,
        rng: &mut u64,
        greedy: bool,
        excluded: i32,
    ) -> Result<i32> {
        if !self.ready || self.poisoned {
            return Err(asset_error("V4.1 logits unavailable"));
        }
        if self
            .logits
            .iter()
            .any(|x| x.is_nan() || *x == f32::INFINITY)
            || !self.logits.iter().any(|x| x.is_finite())
        {
            self.poisoned = true;
            return Err(asset_error("V4.1 logits contain invalid probabilities"));
        }
        if face.dry_multiplier > 0. && self.breakers.is_none() {
            self.breakers = Some(
                (0..self.model.vocab().n_vocab())
                    .map(|t| {
                        self.model
                            .token_text(t)
                            .map(|b| b.iter().any(|x| matches!(x, b'\n' | b'"' | b'*')))
                            .unwrap_or(false)
                    })
                    .collect(),
            );
        }
        let mut row = self.logits.clone();
        penalize(&mut row, &self.generated, self.breakers.as_deref(), face);
        let sampler = if greedy {
            Sampler::GREEDY
        } else {
            Sampler {
                temperature: face.temperature,
                top_k: face.top_k,
                top_p: face.top_p,
                min_p: face.min_p,
            }
        };
        let token = draw(&row, &sampler, rng, [excluded, -1]);
        if token < 0 {
            Err(asset_error("V4.1 sampler has no allowed token"))
        } else {
            Ok(token)
        }
    }
    /// Only committed/emitted tokens enter repetition history, not prompts.
    pub fn record(&mut self, token: i32) {
        self.generated.push(token);
    }
}
impl Drop for V41Request<'_> {
    fn drop(&mut self) {
        // SAFETY: bridge drains pending Engram I/O before releasing native
        // buffers; the boxed provider and image patches are still alive here.
        unsafe {
            ffi::ds4_bridge_v41_request_close(self.raw.as_ptr());
        }
    }
}

/// YoungAi core_decode_penalty.c / ds4_ds41_sample.inc, with scratch owned by
/// the call instead of process-global seen/best arrays. Generated text only.
fn penalize(row: &mut [f32], gen: &[i32], breakers: Option<&[bool]>, f: V41Sampling) {
    let mut seen = std::collections::HashSet::new();
    for &t in gen {
        if let Some(x) = usize::try_from(t).ok().and_then(|i| row.get_mut(i)) {
            *x -= f.freq_penalty;
            if seen.insert(t) {
                *x -= f.presence_penalty;
            }
        }
    }
    let Some(breakers) = breakers else {
        return;
    };
    if f.dry_multiplier <= 0. || f.dry_base <= 1. || f.dry_allowed_length < 1 || gen.len() < 2 {
        return;
    }
    let is_break = |t: i32| {
        usize::try_from(t)
            .ok()
            .and_then(|i| breakers.get(i))
            .copied()
            .unwrap_or(true)
    };
    let last = gen[gen.len() - 1];
    if is_break(last) {
        return;
    }
    let mut best = std::collections::HashMap::<usize, usize>::new();
    for j in 1..gen.len() {
        let t = gen[j];
        if gen[j - 1] != last || is_break(t) {
            continue;
        }
        let mut len = 1;
        while len < 512
            && j > len
            && gen[j - 1 - len] == gen[gen.len() - 1 - len]
            && !is_break(gen[gen.len() - 1 - len])
        {
            len += 1;
        }
        best.entry(t as usize)
            .and_modify(|n| *n = (*n).max(len))
            .or_insert(len);
    }
    for (t, len) in best {
        if len >= f.dry_allowed_length as usize {
            if let Some(x) = row.get_mut(t) {
                *x -= f.dry_multiplier
                    * f.dry_base
                        .powf((len - f.dry_allowed_length as usize).min(64) as f32);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn admission_keeps_reserve_and_rejects_overflow() {
        assert!(admission_fits(4, 12, 4));
        assert!(!admission_fits(4, 11, 4));
        assert!(!admission_fits(u64::MAX, u64::MAX, 0));
        assert!(!admission_fits(1, u64::MAX, u64::MAX));
    }
    #[test]
    fn frequency_and_presence_are_isolated_and_generated_only() {
        let f = V41Sampling {
            freq_penalty: 0.25,
            presence_penalty: 0.5,
            ..Default::default()
        };
        let mut a = [2., 2., 2.];
        let mut b = a;
        penalize(&mut a, &[1, 1, 2], None, f);
        penalize(&mut b, &[0], None, f);
        assert_eq!(a, [2., 1., 1.25]);
        assert_eq!(b, [1.25, 2., 2.]);
    }
    #[test]
    fn dry_breakers_stop_sequence_matching() {
        let f = V41Sampling {
            dry_multiplier: 1.,
            dry_base: 2.,
            dry_allowed_length: 2,
            ..Default::default()
        };
        let mut a = [10.; 3];
        let mut b = a;
        penalize(&mut a, &[0, 1, 0, 1, 0], Some(&[false; 3]), f);
        penalize(&mut b, &[0, 1, 0, 1, 0], Some(&[true, false, false]), f);
        assert!(a[1] < 10.);
        assert_eq!(b, [10.; 3]);
    }
    #[test]
    fn request_rng_is_independent_of_interleaving() {
        let s = Sampler {
            temperature: 0.6,
            top_k: 0,
            top_p: 0.95,
            min_p: 0.,
        };
        let mut a = 7;
        let mut b = 53;
        let mut serial = 7;
        for _ in 0..128 {
            let want = draw(&[0., 1., 2.], &s, &mut serial, [-1, -1]);
            let got = draw(&[0., 1., 2.], &s, &mut a, [-1, -1]);
            let _ = draw(&[2., 1., 0.], &s, &mut b, [-1, -1]);
            assert_eq!(got, want);
        }
    }
}
