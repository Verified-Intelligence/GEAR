# NLP written by GAMS Convert at 03/06/26 16:51:13
#
# Equation counts
#     Total        E        G        L        N        X        C        B
#         2        0        2        0        0        0        0        0
#
# Variable counts
#                  x        b        i      s1s      s2s       sc       si
#     Total     cont   binary  integer     sos1     sos2    scont     sint
#         3        3        0        0        0        0        0        0
# FX      0
#
# Nonzero counts
#     Total    const       NL
#         4        0        4
#
# Reformulation has removed 1 variable and 1 equation

var x1 := 22.3, >= 0, <= 36;
var x2 := 0.5, >= 0, <= 5;
var x3 := 125, >= 0, <= 125;

minimize obj: -2.01e-09 * x1^4 * x2 * x3^2;

subject to

e1:  -x2 * x1^2 >= -675;
e2:  -1e-07 * x1^2 * x3^2 >= -0.419;
