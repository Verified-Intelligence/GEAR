# NLP written by GAMS Convert at 03/06/26 16:51:14
#
# Equation counts
#     Total        E        G        L        N        X        C        B
#         4        2        1        1        0        0        0        0
#
# Variable counts
#                  x        b        i      s1s      s2s       sc       si
#     Total     cont   binary  integer     sos1     sos2    scont     sint
#         3        3        0        0        0        0        0        0
# FX      0
#
# Nonzero counts
#     Total    const       NL
#         7        2        5
#
# Reformulation has removed 1 variable and 1 equation

var x1 >= -50, <= 50;
var x2 >= -50, <= 50;
var x3 >= -50, <= 50;

maximize obj: sqrt((x1 - x3)^2);

subject to

e1:  sqrt(x1^2) - x2 = 0;
e2:  x2 >= 20;
e3:  sqrt((x2 - x3)^2) <= 9;
e4:  sqrt((x1 - x3)^2) = 5;
