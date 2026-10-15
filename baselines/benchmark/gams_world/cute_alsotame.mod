# NLP written by GAMS Convert at 03/06/26 16:50:35
#
# Equation counts
#     Total        E        G        L        N        X        C        B
#         1        1        0        0        0        0        0        0
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

var x1 >= -2, <= 2;
var x2 >= -1.5, <= 1.5;

minimize obj: exp(x1 - 2 * x2);

subject to

e1:  sin(-1 - x1 + x2) = 0;
