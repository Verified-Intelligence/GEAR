# NLP written by GAMS Convert at 03/06/26 16:51:13
#
# Equation counts
#     Total        E        G        L        N        X        C        B
#         3        3        0        0        0        0        0        0
#
# Variable counts
#                  x        b        i      s1s      s2s       sc       si
#     Total     cont   binary  integer     sos1     sos2    scont     sint
#        10       10        0        0        0        0        0        0
# FX      0
#
# Nonzero counts
#     Total    const       NL
#        14       14        0
#
# Reformulation has removed 1 variable and 1 equation

var x1 := 0.1, >= 1e-05, <= 10;
var x2 := 0.1, >= 1e-05, <= 10;
var x3 := 0.1, >= 1e-05, <= 10;
var x4 := 0.1, >= 1e-05, <= 10;
var x5 := 0.1, >= 1e-05, <= 10;
var x6 := 0.1, >= 1e-05, <= 10;
var x7 := 0.1, >= 1e-05, <= 10;
var x8 := 0.1, >= 1e-05, <= 10;
var x9 := 0.1, >= 1e-05, <= 10;
var x10 := 0.1, >= 1e-05, <= 10;

minimize obj: x1 * (log(x1 / (x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10))
    - 6.089) + x2 * (log(x2 / (x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 +
    x10)) - 17.164) + x3 * (log(x3 / (x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 +
    x9 + x10)) - 34.054) + x4 * (log(x4 / (x1 + x2 + x3 + x4 + x5 + x6 + x7 +
    x8 + x9 + x10)) - 5.914) + x5 * (log(x5 / (x1 + x2 + x3 + x4 + x5 + x6 + x7
    + x8 + x9 + x10)) - 24.721) + x6 * (log(x6 / (x1 + x2 + x3 + x4 + x5 + x6
    + x7 + x8 + x9 + x10)) - 14.986) + x7 * (log(x7 / (x1 + x2 + x3 + x4 + x5
    + x6 + x7 + x8 + x9 + x10)) - 24.1) + x8 * (log(x8 / (x1 + x2 + x3 + x4 +
    x5 + x6 + x7 + x8 + x9 + x10)) - 10.708) + x9 * (log(x9 / (x1 + x2 + x3 +
    x4 + x5 + x6 + x7 + x8 + x9 + x10)) - 26.662) + x10 * (log(x10 / (x1 + x2
    + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10)) - 22.179);

subject to

e1:  x1 - 2 * x2 + 2 * x3 + x6 + x10 = 2;
e2:  x4 - 2 * x5 + x6 + x7 = 1;
e3:  x3 + x7 + x8 + 2 * x9 + x10 = 1;
