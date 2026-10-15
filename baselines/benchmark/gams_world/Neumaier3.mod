# NLP written by GAMS Convert at 03/06/26 16:51:15
#
# Equation counts
#     Total        E        G        L        N        X        C        B
#         0        0        0        0        0        0        0        0
#
# Variable counts
#                  x        b        i      s1s      s2s       sc       si
#     Total     cont   binary  integer     sos1     sos2    scont     sint
#        10       10        0        0        0        0        0        0
# FX      0
#
# Nonzero counts
#     Total    const       NL
#         0        0        0
#
# Reformulation has removed 1 variable and 1 equation

var x1 >= -100, <= 100;
var x2 >= -100, <= 100;
var x3 >= -100, <= 100;
var x4 >= -100, <= 100;
var x5 >= -100, <= 100;
var x6 >= -100, <= 100;
var x7 >= -100, <= 100;
var x8 >= -100, <= 100;
var x9 >= -100, <= 100;
var x10 >= -100, <= 100;

minimize obj: (-1 + x1)^2 + (-1 + x2)^2 + (-1 + x3)^2 + (-1 + x4)^2 + (-1 + x5)
    ^2 + (-1 + x6)^2 + (-1 + x7)^2 + (-1 + x8)^2 + (-1 + x9)^2 + (-1 + x10)^2
    + (-x2 * x1) - x3 * x2 - x4 * x3 - x5 * x4 - x6 * x5 - x7 * x6 - x8 * x7
    - x9 * x8 - x10 * x9;

