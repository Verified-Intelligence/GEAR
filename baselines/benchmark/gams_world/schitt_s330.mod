# NLP written by GAMS Convert at 03/06/26 16:51:13
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

var x1 := 2.5, >= 1e-07, <= 5;
var x2 := 2.5, >= 1e-07, <= 5;

minimize obj: 0.1 * (0.44 * x1^3 / x2^2 + 10 / x1 + (0.592 * x1) / x2^3);

subject to

e1:  -8.62 * x2^3 / x1 >= -1;
