# NLP written by GAMS Convert at 03/06/26 16:50:55
#
# Equation counts
#     Total        E        G        L        N        X        C        B
#         1        1        0        0        0        0        0        0
#
# Variable counts
#                  x        b        i      s1s      s2s       sc       si
#     Total     cont   binary  integer     sos1     sos2    scont     sint
#         6        6        0        0        0        0        0        0
# FX      0
#
# Nonzero counts
#     Total    const       NL
#         2        2        0
#
# Reformulation has removed 1 variable and 1 equation

var x1 := -0.5, >= -1.25, <= 1.25;
var x2 := 0.5, >= -11, <= 9;
var x3 := 0.285714285714286, >= -0.285714285714286, <= 1.14285714285714;
var x4 := -0.16, >= -0.2, <= 0.2;
var x5 := 0.04, >= -20.02, <= 19.98;
var x6 := -0.1, >= -0.2, <= 0.2;

minimize obj: 1.04166666666667 * (x1^2 + 0.4 * x1 * x2 + x2^2) + x3^2 + x4^2 +
    x5^2 + x6^2;

subject to

e1:  x1 + 0.5 * x2 = 0.45;
