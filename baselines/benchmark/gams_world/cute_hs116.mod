# NLP written by GAMS Convert at 03/06/26 16:50:56
#
# Equation counts
#     Total        E        G        L        N        X        C        B
#        15        0       14        1        0        0        0        0
#
# Variable counts
#                  x        b        i      s1s      s2s       sc       si
#     Total     cont   binary  integer     sos1     sos2    scont     sint
#        13       13        0        0        0        0        0        0
# FX      0
#
# Nonzero counts
#     Total    const       NL
#        46       15       31
#
# Reformulation has removed 1 variable and 1 equation

var x1 := 0.5, >= 0.1, <= 1;
var x2 := 0.8, >= 0.1, <= 1;
var x3 := 0.9, >= 0.1, <= 1;
var x4 := 0.1, >= 0.0001, <= 0.1;
var x5 := 0.14, >= 0.1, <= 0.9;
var x6 := 0.5, >= 0.1, <= 0.9;
var x7 := 489, >= 0.1, <= 1000;
var x8 := 80, >= 0.1, <= 1000;
var x9 := 650, >= 500, <= 1000;
var x10 := 450, >= 0.1, <= 500;
var x11 := 150, >= 1, <= 150;
var x12 := 150, >= 0.0001, <= 150;
var x13 := 150, >= 0.0001, <= 150;

minimize obj: x11 + x12 + x13;

subject to

e1:  -x2 + x3 >= 0;
e2:  -x1 + x2 >= 0;
e3:  -0.002 * x7 + 0.002 * x8 >= -1;
e4:  x11 + x12 + x13 >= 50;
e5:  1.231059 * x3 * x10 - 1.262626 * x10 + x13 >= 0;
e6:  -0.975 * x2 * x5 - 0.03475 * x2 + x5 + 0.00975 * x2^2 >= 0;
e7:  -0.975 * x3 * x6 - 0.03475 * x3 + x6 + 0.00975 * x3^2 >= 0;
e8:  -0.975 * x1 * x4 - 0.03475 * x1 + x4 + 0.00975 * x1^2 >= 0;
e9:  1.231059 * x2 * x9 - 1.262626 * x9 + x12 >= 0;
e10:  1.231059 * x1 * x8 - 1.262626 * x8 + x11 >= 0;
e11:  x5 * x7 - x1 * x8 - x4 * x7 + x4 * x8 >= 0;
e12:  -0.002 * (x2 * x9 + x5 * x8 - x1 * x8 - x6 * x9) - x5 - x6 >= -1;
e13:  x2 * x9 - x3 * x10 - x6 * x9 - 500 * x2 + 500 * x6 + x2 * x10 >= 0;
e14:  -0.002 * (x2 * x10 - x3 * x10) + x2 >= 0.9;
e15:  x11 + x12 + x13 <= 250;
