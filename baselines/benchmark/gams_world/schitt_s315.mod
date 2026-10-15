# NLP written by GAMS Convert at 03/06/26 16:51:13
#
# Equation counts
#     Total        E        G        L        N        X        C        B
#         3        0        3        0        0        0        0        0
#
# Variable counts
#                  x        b        i      s1s      s2s       sc       si
#     Total     cont   binary  integer     sos1     sos2    scont     sint
#         2        2        0        0        0        0        0        0
# FX      0
#
# Nonzero counts
#     Total    const       NL
#         6        2        4
#
# Reformulation has removed 1 variable and 1 equation

var x1 := -0.1, >= -1, <= 1;
var x2 := -0.9, >= -1, <= 1;

minimize obj: -x2;

subject to

e1:  x1 - 2 * x2 >= -1;
e2:  -x1^2 - x2^2 >= -1;
e3:  x1^2 + x2^2 >= 0;
