# NLP written by GAMS Convert at 03/06/26 16:50:56
#
# Equation counts
#     Total        E        G        L        N        X        C        B
#        10        4        0        6        0        0        0        0
#
# Variable counts
#                  x        b        i      s1s      s2s       sc       si
#     Total     cont   binary  integer     sos1     sos2    scont     sint
#         7        7        0        0        0        0        0        0
# FX      0
#
# Nonzero counts
#     Total    const       NL
#        33       12       21
#
# Reformulation has removed 1 variable and 1 equation

var x1 >= 0, <= 1;
var x2 >= -1, <= 1;
var x3 := 1, >= -1, <= 1;
var x4 := 1, >= -1, <= 1;
var x5 >= -1, <= 1;
var x6 := 1, >= -1, <= 1;
var x7 >= -1, <= 1;

minimize obj: x2^2 + x3^2 + x4^2 + x5^2 + x6^2 + x7^2 + x1^2;

subject to

e1:  x2 + x3 + x4 + x5 + x6 + x7 = -1;
e2:  (1 + x2 + x3 + x4 + x5 + x6 + x7)^2 = 0;
e3:  (6 + x3 + 2 * x4 + 3 * x5 + 4 * x6 + 5 * x7)^2 = 0;
e4:  (30 + 4 * x4 + 9 * x5 + 16 * x6 + 25 * x7)^2 = 0;
e5:  0.001 * (x2^3 - x2) - x1 <= 0;
e6:  0.001 * (x3^3 - x3) - x1 <= 0;
e7:  0.001 * (x4^3 - x4) - x1 <= 0;
e8:  0.001 * (x5^3 - x5) - x1 <= 0;
e9:  0.001 * (x6^3 - x6) - x1 <= 0;
e10:  0.001 * (x7^3 - x7) - x1 <= 0;
