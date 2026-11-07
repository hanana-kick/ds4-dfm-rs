//! Multi-request DSpark: native draft/verify, request-local target sampling.
//! Draft proposals are deterministic. Drawing from the target distribution and
//! accepting only matching proposals preserves that target distribution,
//! including penalties/forced DSML positions. No unverified token is emitted.
use super::*;

pub(super) const VERIFY_MAX: usize = 9;

impl<'m> V41Batch<'m> {
    pub fn enable_mtp(&mut self, verify_k: i32, graph: bool) -> Result<()> {
        if !(0..VERIFY_MAX as i32).contains(&verify_k) {
            return Err(asset_error("invalid V4.1 verification width"));
        }
        // SAFETY: immutable model metadata; no allocation or retained borrow.
        let bytes = unsafe { ffi::ds4_bridge_v41_request_mtp_bytes(self.model.raw_ptr()) };
        if bytes == u64::MAX {
            return Err(asset_error("V4.1 embedded MTP weights/metadata are unavailable"));
        }
        // Immutable owner setting, established before requests are admitted.
        unsafe { ffi::ds4_bridge_v41_set_graph(i32::from(graph)); }
        self.mtp = true;
        self.verify_k = verify_k;
        Ok(())
    }

    /// Run one ragged verification batch for all ready requests. A request
    /// must settle its previous verification before it can enter another one.
    pub fn verify(&mut self, jobs: &mut [(&mut V41Request<'m>, i32, usize)]) -> Result<usize> {
        if self.poisoned || jobs.is_empty() || jobs.len() > self.width {
            return Err(asset_error("invalid/poisoned V4.1 verification cohort"));
        }
        for (r, token, remaining) in jobs.iter() {
            if !r.ready || r.poisoned || !r.verify_ids.is_empty()
                || !std::ptr::eq(r.model, self.model)
                || *token < 0 || *token >= self.model.vocab().n_vocab()
                || *remaining == 0 || *remaining > u32::MAX as usize
            {
                return Err(asset_error("invalid V4.1 verification state/token/budget"));
            }
        }
        let handles: Vec<_> = jobs.iter().map(|(r, _, _)| r.raw.as_ptr()).collect();
        let tokens: Vec<_> = jobs.iter().map(|(_, t, _)| *t).collect();
        let budgets: Vec<_> = jobs.iter().map(|(_, _, n)| *n as u32).collect();
        let mut candidates = vec![0i32; jobs.len() * VERIFY_MAX];
        let mut widths = vec![0u32; jobs.len()];
        let vocab = self.model.vocab().n_vocab() as usize;
        let count = jobs.len() * VERIFY_MAX * vocab;
        let mut err = [0u8; 512];
        // SAFETY: exclusive request references, same immutable model, checked
        // widths, padded output buffers. Native synchronizes before returning.
        let rc = unsafe {
            ffi::ds4_bridge_v41_batch_verify(
                self.raw.as_ptr(), handles.as_ptr(), tokens.as_ptr(), budgets.as_ptr(),
                jobs.len() as i32, candidates.as_mut_ptr(), widths.as_mut_ptr(),
                self.logits.as_mut_ptr(), count, err.as_mut_ptr().cast(), err.len(),
            )
        };
        if rc != 0 || widths.iter().any(|&n| n == 0 || n as usize > VERIFY_MAX) {
            self.poisoned = true;
            for (r, _, _) in jobs.iter_mut() { r.poisoned = true; }
            return Err(if rc != 0 { ffi_error(&err) } else { asset_error("invalid native verify width") });
        }
        let mut total = 0;
        for (i, (r, _, remaining)) in jobs.iter_mut().enumerate() {
            let n = widths[i] as usize;
            if n > *remaining {
                self.poisoned = true;
                r.poisoned = true;
                return Err(asset_error("native verification exceeded output budget"));
            }
            r.logits.clear();
            let at = i * VERIFY_MAX * vocab;
            r.logits.extend_from_slice(&self.logits[at..at + n * vocab]);
            r.verify_ids.clear();
            r.verify_ids.extend_from_slice(&candidates[i * VERIFY_MAX..i * VERIFY_MAX + n]);
            r.verify_row = 0;
            total += n;
        }
        Ok(total)
    }
}

impl V41Request<'_> {
    pub fn verify_rows(&self) -> usize { self.verify_ids.len() }
    pub fn verify_candidate(&self, row: usize) -> Option<i32> { self.verify_ids.get(row).copied() }
    pub fn select_verify_row(&mut self, row: usize) -> Result<()> {
        if self.poisoned || row >= self.verify_ids.len() {
            return Err(asset_error("invalid verification logit row"));
        }
        self.verify_row = row;
        Ok(())
    }

    /// Commit consumed inputs, not the bonus/correction token. Drains Engram
    /// before rewinding; rejected drafts are removed from its token history.
    pub fn commit_verify(&mut self, keep: usize) -> Result<()> {
        let rows = self.verify_ids.len();
        if self.poisoned || keep == 0 || keep > rows {
            return Err(asset_error("invalid V4.1 verification commit"));
        }
        let mut err = [0u8; 512];
        // SAFETY: exclusive request; bridge drains before native rollback.
        let rc = unsafe {
            ffi::ds4_bridge_v41_request_commit(self.raw.as_ptr(), keep as u32,
                err.as_mut_ptr().cast(), err.len())
        };
        if rc != 0 {
            self.poisoned = true;
            return Err(ffi_error(&err));
        }
        let position = self.position();
        if let Some(feed) = self._feed.as_mut() {
            let mut core = feed.core.lock().map_err(|_| asset_error("Engram history lock poisoned"))?;
            core.hist.truncate(position);
            core.image_mask.truncate(position);
            feed.next = None;
        }
        if rows > 1 {
            self.spec_rounds += 1;
            self.spec_offered += (rows - 1) as u64;
            self.spec_accepted += (keep - 1) as u64;
        }
        self.verify_ids.clear();
        self.verify_row = 0;
        Ok(())
    }

    pub fn spec_stats(&self) -> (u64, u64, u64) {
        (self.spec_rounds, self.spec_offered, self.spec_accepted)
    }
}
