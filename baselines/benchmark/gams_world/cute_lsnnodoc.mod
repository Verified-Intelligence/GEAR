# NLP written by GAMS Convert at 03/06/26 16:50:57
#
# Equation counts
#     Total        E        G        L        N        X        C        B
#         4        4        0        0        0        0        0        0
#
# Variable counts
#                  x        b        i      s1s      s2s       sc       si
#     Total     cont   binary  integer     sos1     sos2    scont     sint
#         5        5        0        0        0        0        0        0
# FX      0
#
# Nonzero counts
#     Total    const       NL
#        10       10        0
#
# Reformulation has removed 1 variable and 1 equation

var x1 := 4, >= 2, <= 4;
var x2 := 8, >= 6, <= 8;
var x3 := 2, >= 0, <= 2;
var x4 := 6, >= -1e+04, <= 6;
var x5 := 8, >= -1e+04, <= 8;

minimize obj: x2 * exp(x1 + x3) + x3^2 * x4^2 + (x3 - x5)^2;

subject to

e1:  x1 + x2 = 10;
e2:  -x1 - x3 + x4 = 0;
e3:  -x2 + x3 + x5 = 0;
e4:  -x4 - x5 = -10;
