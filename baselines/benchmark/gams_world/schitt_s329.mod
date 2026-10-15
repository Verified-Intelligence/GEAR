# NLP written by GAMS Convert at 03/06/26 16:51:13
#
# Equation counts
#     Total        E        G        L        N        X        C        B
#         3        1        2        0        0        0        0        0
#
# Variable counts
#                  x        b        i      s1s      s2s       sc       si
#     Total     cont   binary  integer     sos1     sos2    scont     sint
#         2        2        0        0        0        0        0        0
# FX      0
#
# Nonzero counts
#     Total    const       NL
#         6        0        6
#
# Reformulation has removed 1 variable and 1 equation

var x1 := 14.35, >= 13, <= 16;
var x2 := 8.6, >= 0, <= 15;

minimize obj: (-10 + x1)^3 + (-20 + x2)^3;

subject to

e1:  (-5 + x1)^2 + (-5 + x2)^2 >= 100;
e2:  (-6 + x1)^2 + (-5 + x2)^2 >= 0;
e3:  -(-6 + x1)^2 - (-5 + x2)^2 = -82.81;
