# NLP written by GAMS Convert at 03/06/26 16:51:12
#
# Equation counts
#     Total        E        G        L        N        X        C        B
#         4        0        4        0        0        0        0        0
#
# Variable counts
#                  x        b        i      s1s      s2s       sc       si
#     Total     cont   binary  integer     sos1     sos2    scont     sint
#         2        2        0        0        0        0        0        0
# FX      0
#
# Nonzero counts
#     Total    const       NL
#         8        8        0
#
# Reformulation has removed 1 variable and 1 equation

var x1 := 0.1, >= 0, <= 6;
var x2 := 0.1, >= 0, <= 6;

minimize obj: 2 * x1^2 + x2^2 - 48 * x1 - 40 * x2;

subject to

e1:  x1 + 3 * x2 >= 0;
e2:  -x1 - 3 * x2 >= -18;
e3:  x1 + x2 >= 0;
e4:  -x1 - x2 >= -8;
