# NLP written by GAMS Convert at 03/06/26 16:51:15
#
# Equation counts
#     Total        E        G        L        N        X        C        B
#         0        0        0        0        0        0        0        0
#
# Variable counts
#                  x        b        i      s1s      s2s       sc       si
#     Total     cont   binary  integer     sos1     sos2    scont     sint
#         3        3        0        0        0        0        0        0
# FX      0
#
# Nonzero counts
#     Total    const       NL
#         0        0        0
#
# Reformulation has removed 1 variable and 1 equation

var x1 >= -10, <= 10;
var x2 >= -10, <= 10;
var x3 >= -10, <= 10;

minimize obj: (x1 * x3 / (1 + x1 + x2) - 0.126)^2 + (2 * x1 * x3 / (1 + 2 * x1
    + x2) - 0.219)^2 + (x1 * x3 / (1 + x1 + 2 * x2) - 0.076)^2 + (2 * x1 * x3
    / (1 + 2 * x1 + 2 * x2) - 0.126)^2 + (0.1 * x1 * x3 / (1 + 0.1 * x1) -
    0.186)^2;

