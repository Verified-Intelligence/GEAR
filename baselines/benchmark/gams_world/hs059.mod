# DNLP written by GAMS Convert at 03/06/26 16:51:10
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

var x1 := 75, >= 0, <= 75;
var x2 := 65, >= 0, <= 65;

minimize obj: 0.0020567 * x1^3 + 3.8112 * x1 - 1.0345e-05 * x1^4 - 0.030234 *
    x1 * x2 + 6.8306 * x2 + 0.00128134 * x1^2 * x2 + 2.266e-07 * x1^4 * x2 -
    0.25645 * x2^2 + 0.0034604 * x2^3 - 1.3514e-05 * x2^4 + 28.106 / (1 + x2)
    + 5.2375e-06 * x1^2 * x2^2 + 6.3e-08 * x1^3 * x2^2 - 7e-10 * x1^3 * x2^3
    - 0.0003405 * x2^2 * x1 + 1.6638e-06 * x2^3 * x1 + 2.8673 * exp(0.0005 *
    x1 * x2) - 3.5256e-05 * x1^3 * x2 - 0.12694 * x1^2 - 75.196;

subject to

e1:  x1 * x2 >= 700;
e2:  -0.008 * x1^2 + x2 >= 0;
e3:  (-50 + x2)^2 - 5 * x1 >= -275;
