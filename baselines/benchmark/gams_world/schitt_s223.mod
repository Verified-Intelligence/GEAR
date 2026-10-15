# NLP written by GAMS Convert at 03/06/26 16:51:12
#
# Equation counts
#     Total        E        G        L        N        X        C        B
#         2        0        2        0        0        0        0        0
#
# Variable counts
#                  x        b        i      s1s      s2s       sc       si
#     Total     cont   binary  integer     sos1     sos2    scont     sint
#         2        2        0        0        0        0        0        0
# FX      0
#
# Nonzero counts
#     Total    const       NL
#         3        1        2
#
# Reformulation has removed 1 variable and 1 equation

var x1 := 0.1, >= 0, <= 10;
var x2 := 3.3, >= 0, <= 10;

minimize obj: -x1;

subject to

e1:  -exp(exp(x1)) + x2 >= 0;
e2:  exp(exp(x1)) >= 0;
