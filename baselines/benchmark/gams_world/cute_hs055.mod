# NLP written by GAMS Convert at 03/06/26 16:50:55
#
# Equation counts
#     Total        E        G        L        N        X        C        B
#         6        6        0        0        0        0        0        0
#
# Variable counts
#                  x        b        i      s1s      s2s       sc       si
#     Total     cont   binary  integer     sos1     sos2    scont     sint
#         6        6        0        0        0        0        0        0
# FX      0
#
# Nonzero counts
#     Total    const       NL
#        15       15        0
#
# Reformulation has removed 1 variable and 1 equation

var x1 := 1, >= 0, <= 1;
var x2 := 2, >= 0, <= 1e+18;
var x3 >= 0, <= 1e+18;
var x4 >= 0, <= 1;
var x5 >= 0, <= 1e+18;
var x6 := 2, >= 0, <= 1e+18;

minimize obj: exp(x1 * x4) + x1 + 2 * x2 + 4 * x5;

subject to

e1:  x1 + 2 * x2 + 5 * x5 = 6;
e2:  x1 + x2 + x3 = 3;
e3:  x4 + x5 + x6 = 2;
e4:  x1 + x4 = 1;
e5:  x2 + x5 = 2;
e6:  x3 + x6 = 2;
