# NLP written by GAMS Convert at 03/06/26 16:50:55
#
# Equation counts
#     Total        E        G        L        N        X        C        B
#         2        1        1        0        0        0        0        0
#
# Variable counts
#                  x        b        i      s1s      s2s       sc       si
#     Total     cont   binary  integer     sos1     sos2    scont     sint
#         4        4        0        0        0        0        0        0
# FX      0
#
# Nonzero counts
#     Total    const       NL
#         8        0        8
#
# Reformulation has removed 1 variable and 1 equation

var x1 := 1, >= 1, <= 5;
var x2 := 5, >= 1, <= 5;
var x3 := 5, >= 1, <= 5;
var x4 := 1, >= 1, <= 5;

minimize obj: x1 * x4 * (x1 + x2 + x3) + x3;

subject to

e1:  x1 * x2 * x3 * x4 >= 25;
e2:  x1^2 + x2^2 + x3^2 + x4^2 = 40;
