# NLP written by GAMS Convert at 03/06/26 16:51:09
#
# Equation counts
#     Total        E        G        L        N        X        C        B
#         2        0        0        2        0        0        0        0
#
# Variable counts
#                  x        b        i      s1s      s2s       sc       si
#     Total     cont   binary  integer     sos1     sos2    scont     sint
#         2        2        0        0        0        0        0        0
# FX      0
#
# Nonzero counts
#     Total    const       NL
#         4        0        4
#
# Reformulation has removed 1 variable and 1 equation

var x1 := 0.1, >= 0, <= 10;
var x2 >= 0, <= 10;

minimize obj: 7 * x1^2 + 3 * x2^2 - 84 * x1 - 24 * x2 + 300;

subject to

e1:  -x1 * x2 <= -1;
e2:  x1^2 + x2^2 <= 9;
