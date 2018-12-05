#ifndef C_SPARSE_H
#define C_SPARSE_H
#include <libdnn/mem.h>
#include <libfixed/fixed.h>

#define C_SPARSE_LEN 18

__ro_hifram fixed c_sparse[18] = {
    F_LIT(-5), F_LIT(4),  F_LIT(-4), F_LIT(-3), F_LIT(-3), F_LIT(-5),
    F_LIT(2),  F_LIT(4),  F_LIT(-2), F_LIT(2),  F_LIT(-3), F_LIT(-4),
    F_LIT(-2), F_LIT(-3), F_LIT(-2), F_LIT(-2), F_LIT(-3), F_LIT(-3)};

__ro_hifram fixed c_sparse_offsets[18] = {0, 3, 1, 1, 1, 2, 0, 1, 1,
                                          1, 2, 3, 0, 2, 2, 1, 1, 1};

__ro_hifram fixed c_sparse_sizes[3] = {6, 6, 6};

#endif