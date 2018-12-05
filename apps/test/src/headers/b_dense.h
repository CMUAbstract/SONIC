#ifndef B_DENSE_H
#define B_DENSE_H
#include <libdnn/mem.h>
#include <libfixed/fixed.h>

__ro_hifram fixed b_dense[10] = {F_LIT(3),  F_LIT(0),  F_LIT(0),  F_LIT(-3),
                                 F_LIT(0),  F_LIT(-4), F_LIT(-5), F_LIT(-5),
                                 F_LIT(-3), F_LIT(4)};

#endif