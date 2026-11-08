/* SPDX-License-Identifier: MIT
 * Packed MTP round bookkeeping. No allocator, GPU or process-global state.
 * One verification row consumes one already known token. A row's target
 * sample may accept the next draft, or become the next round's pending token.
 */
#ifndef DS41_ROUND_H
#define DS41_ROUND_H
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#define DS41_VERIFY_MAX 9u

typedef struct {
    uint32_t rows, cursor;
    int32_t input[DS41_VERIFY_MAX];
} ds41_round;

static inline bool ds41_round_begin(ds41_round *r, const int32_t *input, uint32_t rows) {
    if (!r || !input || rows < 1 || rows > DS41_VERIFY_MAX || r->rows) return false;
    for (uint32_t i = 0; i < rows; i++) r->input[i] = input[i];
    r->cursor = 0; r->rows = rows;
    return true;
}
/* Advance only after the host has applied grammar, penalties, sampling and
 * stop/output-budget checks to this target sample. The bonus is never fed. */
static inline bool ds41_round_accept(ds41_round *r, int32_t target) {
    if (!r || !r->rows || r->cursor + 1 >= r->rows ||
        target != r->input[r->cursor + 1]) return false;
    r->cursor++;
    return true;
}
static inline uint32_t ds41_round_keep(const ds41_round *r) {
    return r && r->rows ? 1u + r->cursor : 0u;
}
static inline void ds41_round_clear(ds41_round *r) {
    if (r) { r->rows = 0; r->cursor = 0; }
}
#endif
