/* SPDX-License-Identifier: MIT
 * Bounded packed verification layout. Shared by production and CPU tests.
 * Sequences and verification rows are different capacities.
 */
#ifndef DS41_SPEC_PLAN_H
#define DS41_SPEC_PLAN_H
#include <stddef.h>
#include <stdint.h>
#define DS41_VERIFY_MAX 9u
static inline uint32_t ds41_verify_width(uint32_t drafts, uint32_t remaining,
                                        uint32_t room, uint32_t limit) {
    if (!remaining || !room) return 0;
    if (drafts >= DS41_VERIFY_MAX) drafts = DS41_VERIFY_MAX - 1u;
    if (limit && drafts > limit) drafts = limit;
    uint32_t n = drafts + 1u;
    if (n > remaining) n = remaining;
    if (n > room) n = room;
    return n;
}
static inline int ds41_verify_layout(const uint32_t *rows, uint32_t count,
                                     uint32_t max_sequences, uint32_t capacity,
                                     uint32_t *offsets, uint32_t *total) {
    if (!rows || !offsets || !total || !count || count > max_sequences) return 0;
    uint32_t sum = 0;
    for (uint32_t i = 0; i < count; ++i) {
        if (!rows[i] || rows[i] > DS41_VERIFY_MAX || sum > capacity ||
            rows[i] > capacity - sum) return 0;
        sum += rows[i];
    }
    uint32_t off = 0;
    for (uint32_t i = 0; i < count; ++i) { offsets[i] = off; off += rows[i]; }
    *total = sum;
    return 1;
}
#endif
