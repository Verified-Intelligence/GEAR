# NLP written by GAMS Convert at 03/06/26 16:51:05
#
# Equation counts
#     Total        E        G        L        N        X        C        B
#         1        0        0        1        0        0        0        0
#
# Variable counts
#                  x        b        i      s1s      s2s       sc       si
#     Total     cont   binary  integer     sos1     sos2    scont     sint
#         2        2        0        0        0        0        0        0
# FX      0
#
# Nonzero counts
#     Total    const       NL
#         2        2        0
#
# Reformulation has removed 1 variable and 1 equation

var x1 := 0.03, >= 0, <= 1;
var x2 := 0.1, >= 0, <= 1;

minimize obj: -1.5 * (1 - x2)^2 + 1.25 * ((1 - x2)^2 + (1 - x1)^2) + 3 * x1 + 3
    * (0.0416666666666667 * x1^3 + (-0.25 * (1 - x2)^2 + 0.25) * (1 - x1) +
    0.5 * (1 - x2)^2 * (1 - 0.5 * x1 - 0.5 * x2) - 0.166666666666667 * (1 - x2)
    ^3) - 4.5;

subject to

e1:  x1 - x2 <= 0;
