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

var x1 >= -2, <= 2;
var x2 >= -2, <= 2;

minimize obj: -0.5 * exp(-100 * (x1^2 + x2^2)) - 1.2 * exp(-4 * ((-1 + x1)^2 +
    x2^2)) - exp(-4 * (x1^2 + (0.5 + x2)^2)) - exp(-4 * ((0.5 + x1)^2 + x2^2))
    - 1.2 * exp(-4 * (x1^2 + (-1 + x2)^2));

