#ifndef B_SPARSE_H
#define B_SPARSE_H
#include <libdnn/mem.h>
#include <libfixed/fixed.h>

#define B_SPARSE_LEN 7

__ro_hifram fixed b_sparse[7] = {F_LIT(3),  F_LIT(-3), F_LIT(-4), F_LIT(-5),
                                 F_LIT(-5), F_LIT(-3), F_LIT(4)};

__ro_hifram uint16_t b_sparse_offsets[7] = {0, 0, 0, 0, 0, 0, 0};

__ro_hifram uint16_t b_sparse_sizes[11] = {0, 1, 1, 1, 2, 2, 3, 4, 5, 6, 7};

#endif