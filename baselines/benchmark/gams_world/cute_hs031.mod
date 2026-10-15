# NLP written by GAMS Convert at 03/06/26 16:50:55
#
# Equation counts
#     Total        E        G        L        N        X        C        B
#         1        0        1        0        0        0        0        0
#
# Variable counts
#                  x        b        i      s1s      s2s       sc       si
#     Total     cont   binary  integer     sos1     sos2    scont     sint
#         3        3        0        0        0        0        0        0
# FX      0
#
# Nonzero counts
#     Total    const       NL
#         2        0        2
#
# Reformulation has removed 1 variable and 1 equation

var x1 := 1, >= -10, <= 10;
var x2 := 1, >= 1, <= 10;
var x3 := 1, >= -10, <= 1;

minimize obj: 9 * x1^2 + x2^2 + 9 * x3^2;

subject to

e1:  x1 * x2 >= 1;
