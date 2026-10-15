# NLP written by GAMS Convert at 03/06/26 16:50:41
#
# Equation counts
#     Total        E        G        L        N        X        C        B
#         2        0        1        1        0        0        0        0
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

var x1 := 20.1, >= 13, <= 100;
var x2 := 5.84, >= 0, <= 50;

minimize obj: (-10 + x1)^3 + (-20 + x2)^3;

subject to

e1:  (-5 + x1)^2 + (-5 + x2)^2 >= 100;
e2:  (-5 + x2)^2 + (-6 + x1)^2 <= 82.81;
