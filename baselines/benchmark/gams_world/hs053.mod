# NLP written by GAMS Convert at 03/06/26 16:51:10
#
# Equation counts
#     Total        E        G        L        N        X        C        B
#         3        3        0        0        0        0        0        0
#
# Variable counts
#                  x        b        i      s1s      s2s       sc       si
#     Total     cont   binary  integer     sos1     sos2    scont     sint
#         5        5        0        0        0        0        0        0
# FX      0
#
# Nonzero counts
#     Total    const       NL
#         7        7        0
#
# Reformulation has removed 1 variable and 1 equation

var x1 := 2, >= -10, <= 10;
var x2 := 2, >= -10, <= 10;
var x3 := 2, >= -10, <= 10;
var x4 := 2, >= -10, <= 10;
var x5 := 2, >= -10, <= 10;

minimize obj: (x1 - x2)^2 + (-2 + x2 + x3)^2 + (-1 + x4)^2 + (-1 + x5)^2;

subject to

e1:  x1 + 3 * x2 = 0;
e2:  x3 + x4 - 2 * x5 = 0;
e3:  x2 - x5 = 0;
