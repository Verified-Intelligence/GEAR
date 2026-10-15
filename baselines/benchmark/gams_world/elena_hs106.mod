# NLP written by GAMS Convert at 03/06/26 16:51:11
#
# Equation counts
#     Total        E        G        L        N        X        C        B
#         6        0        6        0        0        0        0        0
#
# Variable counts
#                  x        b        i      s1s      s2s       sc       si
#     Total     cont   binary  integer     sos1     sos2    scont     sint
#         8        8        0        0        0        0        0        0
# FX      0
#
# Nonzero counts
#     Total    const       NL
#        17        9        8
#
# Reformulation has removed 1 variable and 1 equation

var x1 := 5000, >= 100, <= 1e+04;
var x2 := 5000, >= 1000, <= 1e+04;
var x3 := 5000, >= 1000, <= 1e+04;
var x4 := 200, >= 10, <= 1000;
var x5 := 350, >= 10, <= 1000;
var x6 := 150, >= 10, <= 1000;
var x7 := 225, >= 10, <= 1000;
var x8 := 425, >= 10, <= 1000;

minimize obj: x1 + x2 + x3;

subject to

e1:  -0.0025 * x4 - 0.0025 * x6 >= -1;
e2:  0.0025 * x4 - 0.0025 * x5 - 0.0025 * x7 >= -1;
e3:  0.01 * x5 - 0.01 * x8 >= -1;
e4:  x1 * x6 - 100 * x1 - 833.3325 * x4 >= -83333.33;
e5:  x2 * x7 - x2 * x4 + 1250 * x4 - 1250 * x5 >= 0;
e6:  x3 * x8 - x3 * x5 + 2500 * x5 >= 1.25e+06;
