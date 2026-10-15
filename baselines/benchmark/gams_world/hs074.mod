# NLP written by GAMS Convert at 03/06/26 16:51:10
#
# Equation counts
#     Total        E        G        L        N        X        C        B
#         5        3        1        1        0        0        0        0
#
# Variable counts
#                  x        b        i      s1s      s2s       sc       si
#     Total     cont   binary  integer     sos1     sos2    scont     sint
#         4        4        0        0        0        0        0        0
# FX      0
#
# Nonzero counts
#     Total    const       NL
#        12        6        6
#
# Reformulation has removed 1 variable and 1 equation

var x1 >= 0, <= 1200;
var x2 >= 0, <= 1200;
var x3 >= -0.55, <= 0.55;
var x4 >= -0.55, <= 0.55;

minimize obj: 1e-06 * x1^3 + 3 * x1 + 6.66666666666667e-07 * x2^3 + 2 * x2;

subject to

e1:  x3 - x4 <= 0.55;
e2:  x3 - x4 >= -0.55;
e3:  -1000 * sin(-0.25 - x3) - 1000 * sin(-0.25 - x4) + x1 = 894.8;
e4:  -1000 * sin(-0.25 + x3) - 1000 * sin(-0.25 + x3 - x4) + x2 = 894.8;
e5:  -1000 * sin(-0.25 + x4) - 1000 * sin(-0.25 - x3 + x4) = 1294.8;
