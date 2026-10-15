# NLP written by GAMS Convert at 03/06/26 16:50:41
#
# Equation counts
#     Total        E        G        L        N        X        C        B
#         0        0        0        0        0        0        0        0
#
# Variable counts
#                  x        b        i      s1s      s2s       sc       si
#     Total     cont   binary  integer     sos1     sos2    scont     sint
#         2        2        0        0        0        0        0        0
# FX      0
#
# Nonzero counts
#     Total    const       NL
#         0        0        0
#
# Reformulation has removed 1 variable and 1 equation

var x1 := 95, >= 0, <= 95;
var x2 := 10, >= 0, <= 75;

minimize obj: -0.1269366345 * x1^2 + 0.0020567665 * x1^3 - 1.0345e-05 * x1^4 -
    x2 * (-0.0012813448 * x1^2 + 0.0302344793 * x1 + 3.52599e-05 * x1^3 -
    2.266e-07 * x1^4) - 0.2564581253 * x2^2 + 0.003460403 * x2^3 - 1.35139e-05
    * x2^4 + 28.1064434908 / (1 + x2) - (-5.2375e-06 * x1^2 + 0.0003405462 *
    x1 - 6.3e-09 * x1^3) * x2^2 - (7e-10 * x1^3 - 1.6638e-06 * x1) * x2^3 +
    2.8673112392 * exp(0.0005 * x1 * x2) + 3.8112755343 * x1 + 6.8306567613 *
    x2 - 75.1963666677;

