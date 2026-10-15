# NLP written by GAMS Convert at 03/06/26 16:51:13
#
# Equation counts
#     Total        E        G        L        N        X        C        B
#         2        2        0        0        0        0        0        0
#
# Variable counts
#                  x        b        i      s1s      s2s       sc       si
#     Total     cont   binary  integer     sos1     sos2    scont     sint
#         4        4        0        0        0        0        0        0
# FX      0
#
# Nonzero counts
#     Total    const       NL
#         4        4        0
#
# Reformulation has removed 1 variable and 1 equation

var x1 >= 0, <= 1;
var x2 >= 0, <= 1;
var x3 >= 0, <= 1;
var x4 >= 0, <= 1;

minimize obj: -exp(-10 * exp(-x3) * x1) - exp(-10 * exp(-x4) * x2) + 2;

subject to

e1:  x1 + x2 = 1;
e2:  x3 + x4 = 1;
