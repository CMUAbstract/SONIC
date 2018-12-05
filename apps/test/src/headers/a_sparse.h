#ifndef A_SPARSE_H
#define A_SPARSE_H
#include <libdnn/mem.h>
#include <libfixed/fixed.h>

#define A_SPARSE_LEN 99

__ro_hifram fixed a_sparse[99] = {
    F_LIT(2),  F_LIT(-3), F_LIT(3),  F_LIT(-5), F_LIT(-3), F_LIT(3),  F_LIT(2),
    F_LIT(-2), F_LIT(2),  F_LIT(-4), F_LIT(2),  F_LIT(3),  F_LIT(-3), F_LIT(-3),
    F_LIT(4),  F_LIT(-2), F_LIT(3),  F_LIT(2),  F_LIT(-5), F_LIT(4),  F_LIT(-4),
    F_LIT(3),  F_LIT(-4), F_LIT(3),  F_LIT(-2), F_LIT(-5), F_LIT(4),  F_LIT(-2),
    F_LIT(-2), F_LIT(-5), F_LIT(-4), F_LIT(3),  F_LIT(4),  F_LIT(4),  F_LIT(3),
    F_LIT(-5), F_LIT(3),  F_LIT(-3), F_LIT(-4), F_LIT(4),  F_LIT(-3), F_LIT(3),
    F_LIT(4),  F_LIT(2),  F_LIT(-4), F_LIT(-5), F_LIT(3),  F_LIT(-3), F_LIT(-2),
    F_LIT(4),  F_LIT(3),  F_LIT(-4), F_LIT(-5), F_LIT(4),  F_LIT(4),  F_LIT(-4),
    F_LIT(-2), F_LIT(4),  F_LIT(2),  F_LIT(4),  F_LIT(2),  F_LIT(3),  F_LIT(-2),
    F_LIT(-3), F_LIT(-4), F_LIT(-5), F_LIT(-2), F_LIT(-4), F_LIT(-4), F_LIT(3),
    F_LIT(-2), F_LIT(-2), F_LIT(-4), F_LIT(2),  F_LIT(3),  F_LIT(-2), F_LIT(-4),
    F_LIT(-5), F_LIT(4),  F_LIT(-2), F_LIT(4),  F_LIT(-3), F_LIT(3),  F_LIT(-5),
    F_LIT(-4), F_LIT(-4), F_LIT(3),  F_LIT(-2), F_LIT(-4), F_LIT(3),  F_LIT(-2),
    F_LIT(3),  F_LIT(-3), F_LIT(2),  F_LIT(-5), F_LIT(3),  F_LIT(-4), F_LIT(3),
    F_LIT(2)};

__ro_hifram uint16_t a_sparse_offsets[99] = {
    0, 2, 3, 4, 5, 6, 8, 9, 0, 1, 2, 3, 5, 7, 8, 9, 0, 4, 5, 6, 7, 8, 9, 0, 1,
    2, 3, 4, 5, 7, 8, 9, 0, 1, 2, 3, 4, 6, 8, 0, 2, 3, 5, 8, 1, 3, 4, 5, 8, 1,
    2, 3, 4, 6, 7, 8, 0, 1, 2, 9, 1, 5, 6, 7, 0, 1, 2, 6, 8, 9, 0, 2, 3, 5, 6,
    7, 0, 1, 3, 4, 6, 7, 8, 9, 0, 1, 2, 3, 4, 6, 8, 9, 0, 1, 2, 4, 7, 8, 9};

__ro_hifram uint16_t a_sparse_sizes[16] = {0,  8,  16, 23, 32, 39, 44, 49,
                                           56, 60, 64, 70, 76, 84, 92, 99};

#endif