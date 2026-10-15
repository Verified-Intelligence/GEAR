# NLP written by GAMS Convert at 03/06/26 16:51:16
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

var x1 >= -10, <= 10;
var x2 >= -10, <= 10;

minimize obj: (cos(1 + 2 * x1) + 2 * cos(2 + 3 * x1) + 3 * cos(3 + 4 * x1) + 4
    * cos(4 + 5 * x1) + 5 * cos(5 + 6 * x1)) * (cos(1 + 2 * x2) + 2 * cos(2 +
    3 * x2) + 3 * cos(3 + 4 * x2) + 4 * cos(4 + 5 * x2) + 5 * cos(5 + 6 * x2));

