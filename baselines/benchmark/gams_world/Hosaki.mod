# NLP written by GAMS Convert at 03/06/26 16:51:15
#
# Equation counts
#     Total        E        G        L        N        X        C        B
#         0        0        0        0        0        0        0        0
#
# Variable counts
#                  x        b        i      s1s      s2s       sc       si
#     Total     cont   binary  integer     sos1     sos2    scont     sint
#         2        2        0        0        0        0        0        0
# FX      0
#
# Nonzero counts
#     Total    const       NL
#         0        0        0
#
# Reformulation has removed 1 variable and 1 equation

var x1 >= 0, <= 5;
var x2 >= 0, <= 6;

minimize obj: x2 * (7 * x1 * x1 - 8 * x1 - 2.33333333333333 * x1 * x1 * x1 +
    0.25 * x1 * x1 * x1 * x1 + 1) * x2 * exp(-x2);

