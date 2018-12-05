#ifndef A_DENSE_H
#define A_DENSE_H
#include <libdnn/mem.h>
#include <libfixed/fixed.h>

__ro_hifram fixed a_dense[15][10] = {
    {F_LIT(2), F_LIT(0), F_LIT(-3), F_LIT(3), F_LIT(-5), F_LIT(-3), F_LIT(3),
     F_LIT(0), F_LIT(2), F_LIT(-2)},
    {F_LIT(2), F_LIT(-4), F_LIT(2), F_LIT(3), F_LIT(0), F_LIT(-3), F_LIT(0),
     F_LIT(-3), F_LIT(4), F_LIT(-2)},
    {F_LIT(3), F_LIT(0), F_LIT(0), F_LIT(0), F_LIT(2), F_LIT(-5), F_LIT(4),
     F_LIT(-4), F_LIT(3), F_LIT(-4)},
    {F_LIT(3), F_LIT(-2), F_LIT(-5), F_LIT(4), F_LIT(-2), F_LIT(-2), F_LIT(0),
     F_LIT(-5), F_LIT(-4), F_LIT(3)},
    {F_LIT(4), F_LIT(4), F_LIT(3), F_LIT(-5), F_LIT(3), F_LIT(0), F_LIT(-3),
     F_LIT(0), F_LIT(-4), F_LIT(0)},
    {F_LIT(4), F_LIT(0), F_LIT(-3), F_LIT(3), F_LIT(0), F_LIT(4), F_LIT(0),
     F_LIT(0), F_LIT(2), F_LIT(0)},
    {F_LIT(0), F_LIT(-4), F_LIT(0), F_LIT(-5), F_LIT(3), F_LIT(-3), F_LIT(0),
     F_LIT(0), F_LIT(-2), F_LIT(0)},
    {F_LIT(0), F_LIT(4), F_LIT(3), F_LIT(-4), F_LIT(-5), F_LIT(0), F_LIT(4),
     F_LIT(4), F_LIT(-4), F_LIT(0)},
    {F_LIT(-2), F_LIT(4), F_LIT(2), F_LIT(0), F_LIT(0), F_LIT(0), F_LIT(0),
     F_LIT(0), F_LIT(0), F_LIT(4)},
    {F_LIT(0), F_LIT(2), F_LIT(0), F_LIT(0), F_LIT(0), F_LIT(3), F_LIT(-2),
     F_LIT(-3), F_LIT(0), F_LIT(0)},
    {F_LIT(-4), F_LIT(-5), F_LIT(-2), F_LIT(0), F_LIT(0), F_LIT(0), F_LIT(-4),
     F_LIT(0), F_LIT(-4), F_LIT(3)},
    {F_LIT(-2), F_LIT(0), F_LIT(-2), F_LIT(-4), F_LIT(0), F_LIT(2), F_LIT(3),
     F_LIT(-2), F_LIT(0), F_LIT(0)},
    {F_LIT(-4), F_LIT(-5), F_LIT(0), F_LIT(4), F_LIT(-2), F_LIT(0), F_LIT(4),
     F_LIT(-3), F_LIT(3), F_LIT(-5)},
    {F_LIT(-4), F_LIT(-4), F_LIT(3), F_LIT(-2), F_LIT(-4), F_LIT(0), F_LIT(3),
     F_LIT(0), F_LIT(-2), F_LIT(3)},
    {F_LIT(-3), F_LIT(2), F_LIT(-5), F_LIT(0), F_LIT(3), F_LIT(0), F_LIT(0),
     F_LIT(-4), F_LIT(3), F_LIT(2)}};

#endif