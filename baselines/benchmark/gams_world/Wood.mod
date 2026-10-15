# NLP written by GAMS Convert at 03/06/26 16:51:16
#
# Equation counts
#     Total        E        G        L        N        X        C        B
#         0        0        0        0        0        0        0        0
#
# Variable counts
#                  x        b        i      s1s      s2s       sc       si
#     Total     cont   binary  integer     sos1     sos2    scont     sint
#         4        4        0        0        0        0        0        0
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
var x4 >= -10, <= 10;

minimize obj: 100 * (-x1 * x1 + x2)^2 + (1 - x1) * (1 - x1) + 90 * (-x3 * x3 +
    x4)^2 + (1 - x3) * (1 - x3) + 10.1 * ((-1 + x2)^2 + (-1 + x4)^2) + (-19.8
    + 19.8 * x2) * (-1 + x4);

