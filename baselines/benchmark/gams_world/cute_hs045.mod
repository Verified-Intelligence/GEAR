# NLP written by GAMS Convert at 03/06/26 16:50:55
#
# Equation counts
#     Total        E        G        L        N        X        C        B
#         0        0        0        0        0        0        0        0
#
# Variable counts
#                  x        b        i      s1s      s2s       sc       si
#     Total     cont   binary  integer     sos1     sos2    scont     sint
#         5        5        0        0        0        0        0        0
# FX      0
#
# Nonzero counts
#     Total    const       NL
#         0        0        0
#
# Reformulation has removed 1 variable and 1 equation

var x1 >= 0, <= 1;
var x2 >= 0, <= 2;
var x3 >= 0, <= 3;
var x4 >= 0, <= 4;
var x5 >= 0, <= 5;

minimize obj: -0.00833333333333333 * x1 * x2 * x3 * x4 * x5 + 2;

