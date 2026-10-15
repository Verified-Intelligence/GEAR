# NLP written by GAMS Convert at 03/06/26 16:50:41
#
# Equation counts
#     Total        E        G        L        N        X        C        B
#         5        0        5        0        0        0        0        0
#
# Variable counts
#                  x        b        i      s1s      s2s       sc       si
#     Total     cont   binary  integer     sos1     sos2    scont     sint
#         2        2        0        0        0        0        0        0
# FX      0
#
# Nonzero counts
#     Total    const       NL
#        10        4        6
#
# Reformulation has removed 1 variable and 1 equation

var x1 := 3, >= -50, <= 50;
var x2 := 1, >= -50, <= 50;

minimize obj: x1^2 + x2^2;

subject to

e1:  x1 + x2 >= 1;
e2:  x1^2 + x2^2 >= 1;
e3:  9 * x1^2 + x2^2 >= 9;
e4:  x1^2 - x2 >= 0;
e5:  x2^2 - x1 >= 0;
