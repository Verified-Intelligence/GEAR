# NLP written by GAMS Convert at 03/06/26 16:51:15
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

minimize obj: 121 * x1 * x1 + (5 * x3 - 5 * x4) * (x3 - x4) + (x2 - 2 * x3)^4
    + 10 * (x1 - x4)^4;

