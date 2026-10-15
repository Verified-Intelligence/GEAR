# NLP written by GAMS Convert at 03/06/26 16:50:35
#
# Equation counts
#     Total        E        G        L        N        X        C        B
#         1        0        1        0        0        0        0        0
#
# Variable counts
#                  x        b        i      s1s      s2s       sc       si
#     Total     cont   binary  integer     sos1     sos2    scont     sint
#         2        2        0        0        0        0        0        0
# FX      0
#
# Nonzero counts
#     Total    const       NL
#         2        0        2
#
# Reformulation has removed 1 variable and 1 equation

var x1 >= 0, <= 115.8;
var x2 := 1e-05, >= 1e-05, <= 30;

minimize obj: 29.4 * x1 + 18 * x2;

subject to

e1:  -0.2458 * x1^2 / x2 + x1 >= 6;
