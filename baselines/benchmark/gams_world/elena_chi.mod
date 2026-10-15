# NLP written by GAMS Convert at 03/06/26 16:51:11
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

var x1 >= -30, <= 30;
var x2 >= -30, <= 30;

minimize obj: x1^2 - 12 * x1 + 10 * cos(1.5707963267949 * x1) + 8 * sin(
    15.707963267949 * x1) - 0.447213595499958 * exp(-0.5 * (-0.5 + x2)^2) + 11;

