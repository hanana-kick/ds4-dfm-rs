//! Request-local verification cursor, independent of the transport and GPU.
//! The host samples target logits one row at a time. A greedy draft is a
//! point mass: matching target draws are accepted; the first different draw
//! is the bonus token. Never draw from or penalize an unaccepted future row.
pub(super) const VERIFY_MAX: usize = 9;

#[derive(Default)]
pub(super) struct VerifiedRound {
    input: [i32; VERIFY_MAX],
    rows: usize,
    cursor: usize,
}
impl VerifiedRound {
    pub(super) fn begin(&mut self, input: &[i32]) -> bool {
        if self.rows != 0 || input.is_empty() || input.len() > VERIFY_MAX {
            return false;
        }
        self.input[..input.len()].copy_from_slice(input);
        self.rows = input.len();
        self.cursor = 0;
        true
    }
    pub(super) fn rows(&self) -> usize { self.rows }
    pub(super) fn cursor(&self) -> usize { self.cursor }
    pub(super) fn keep(&self) -> usize {
        if self.rows == 0 { 0 } else { self.cursor + 1 }
    }
    pub(super) fn accept(&mut self, target: i32) -> bool {
        let next = self.cursor + 1;
        if self.rows == 0 || next >= self.rows || target != self.input[next] {
            return false;
        }
        self.cursor = next;
        true
    }
    pub(super) fn clear(&mut self) { self.rows = 0; self.cursor = 0; }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn every_rejection_depth_keeps_only_confirmed_inputs() {
        for accepted in 0..=5 {
            let mut r = VerifiedRound::default();
            assert!(r.begin(&[10, 11, 12, 13, 14, 15]));
            for j in 0..accepted { assert!(r.accept(11 + j as i32)); }
            assert!(!r.accept(99));
            assert_eq!(r.keep(), 1 + accepted);
            r.clear(); assert_eq!(r.keep(), 0);
        }
    }
    #[test]
    fn four_requests_accept_different_prefixes_without_cross_talk() {
        let mut all: [VerifiedRound; 4] = std::array::from_fn(|_| VerifiedRound::default());
        for (i, r) in all.iter_mut().enumerate() {
            assert!(r.begin(&[i as i32 * 10, i as i32 * 10 + 1, i as i32 * 10 + 2]));
        }
        assert!(all[0].accept(1)); assert!(all[0].accept(2));
        assert!(!all[1].accept(99));
        assert!(all[2].accept(21)); assert!(!all[2].accept(99));
        all[3].clear(); // cancellation must not mutate a sibling
        assert_eq!(all.map(|r| r.keep()), [3, 1, 2, 0]);
    }
    #[test]
    fn a_bonus_cannot_advance_past_the_last_logit_row() {
        let mut r = VerifiedRound::default();
        assert!(r.begin(&[1, 2])); assert!(r.accept(2));
        assert!(!r.accept(3)); assert_eq!(r.keep(), 2);
        assert!(!r.begin(&[4]));
        r.clear(); assert!(r.begin(&[4])); assert!(!r.accept(5));
    }
    #[test]
    fn eos_budget_and_forced_tool_syntax_stop_before_accept() {
        let mut r = VerifiedRound::default();
        assert!(r.begin(&[10, 11, 12, 13]));
        assert!(r.accept(11));
        // EOS/output cap terminates before accepting row 12. A grammar-forced
        // token different from 12 takes the same mismatch path.
        assert_eq!(r.keep(), 2);
        assert!(!r.accept(77)); assert_eq!(r.keep(), 2);
    }
    #[test]
    fn invalid_width_and_double_begin_do_not_mutate_a_round() {
        let mut r = VerifiedRound::default();
        assert!(!r.begin(&[])); assert!(!r.begin(&[0; VERIFY_MAX + 1]));
        assert!(r.begin(&[1, 2, 3])); assert!(!r.begin(&[99]));
        assert_eq!(r.rows(), 3); assert_eq!(r.cursor(), 0);
        assert!(r.accept(2));
    }
}
