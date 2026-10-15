# DNLP written by GAMS Convert at 03/06/26 16:51:11
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

var x1 := 0.5, >= -2, <= 2;
var x2 := -0.5, >= -2, <= 2;

minimize obj: ((1 + x1 + x2)^2 * (3 * x1^2 - 14 * x1 + 6 * x1 * x2 - 14 * x2 +
    3 * x2^2 + 19) + 1) * ((2 * x1 - 3 * x2)^2 * (12 * x1^2 - 32 * x1 - 36 * x1
    * x2 + 48 * x2 + 27 * x2^2 + 18) + 30);

