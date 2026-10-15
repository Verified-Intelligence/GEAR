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

var x1 >= -5, <= 10;
var x2 >= 0, <= 15;

minimize obj: (-0.129184509143981 * x1^2 + 1.59154943091895 * x1 - 6 + x2)^2 +
    9.60211264227026 * cos(x1) + 10;

