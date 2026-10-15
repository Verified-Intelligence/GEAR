# NLP written by GAMS Convert at 03/06/26 16:51:14
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

minimize obj: -cos(x1) * cos(x2) * exp(-(-3.14159265359 + x1)^2 - (
    -3.14159265359 + x2)^2);

