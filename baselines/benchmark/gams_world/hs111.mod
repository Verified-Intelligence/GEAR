# NLP written by GAMS Convert at 03/06/26 16:51:11
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
#        14        0       14
#
# Reformulation has removed 1 variable and 1 equation

var x1 := -2.3, >= -100, <= 100;
var x2 := -2.3, >= -100, <= 100;
var x3 := -2.3, >= -100, <= 100;
var x4 := -2.3, >= -100, <= 100;
var x5 := -2.3, >= -100, <= 100;
var x6 := -2.3, >= -100, <= 100;
var x7 := -2.3, >= -100, <= 100;
var x8 := -2.3, >= -100, <= 100;
var x9 := -2.3, >= -100, <= 100;
var x10 := -2.3, >= -100, <= 100;

minimize obj: exp(x1) * (-log(exp(x1) + exp(x2) + exp(x3) + exp(x4) + exp(x5)
    + exp(x6) + exp(x7) + exp(x8) + exp(x9) + exp(x10)) + x1 - 6.089) + exp(x2)
    * (-log(exp(x1) + exp(x2) + exp(x3) + exp(x4) + exp(x5) + exp(x6) + exp(x7)
    + exp(x8) + exp(x9) + exp(x10)) + x2 - 17.164) + exp(x3) * (-log(exp(x1)
    + exp(x2) + exp(x3) + exp(x4) + exp(x5) + exp(x6) + exp(x7) + exp(x8) +
    exp(x9) + exp(x10)) + x3 - 34.054) + exp(x4) * (-log(exp(x1) + exp(x2) +
    exp(x3) + exp(x4) + exp(x5) + exp(x6) + exp(x7) + exp(x8) + exp(x9) + exp(
    x10)) + x4 - 5.914) + exp(x5) * (-log(exp(x1) + exp(x2) + exp(x3) + exp(x4)
    + exp(x5) + exp(x6) + exp(x7) + exp(x8) + exp(x9) + exp(x10)) + x5 -
    24.721) + exp(x6) * (-log(exp(x1) + exp(x2) + exp(x3) + exp(x4) + exp(x5)
    + exp(x6) + exp(x7) + exp(x8) + exp(x9) + exp(x10)) + x6 - 14.986) + exp(
    x7) * (-log(exp(x1) + exp(x2) + exp(x3) + exp(x4) + exp(x5) + exp(x6) +
    exp(x7) + exp(x8) + exp(x9) + exp(x10)) + x7 - 24.1) + exp(x8) * (-log(exp(
    x1) + exp(x2) + exp(x3) + exp(x4) + exp(x5) + exp(x6) + exp(x7) + exp(x8)
    + exp(x9) + exp(x10)) + x8 - 10.708) + exp(x9) * (-log(exp(x1) + exp(x2)
    + exp(x3) + exp(x4) + exp(x5) + exp(x6) + exp(x7) + exp(x8) + exp(x9) +
    exp(x10)) + x9 - 26.662) + exp(x10) * (-log(exp(x1) + exp(x2) + exp(x3) +
    exp(x4) + exp(x5) + exp(x6) + exp(x7) + exp(x8) + exp(x9) + exp(x10)) + x10
    - 22.179);

subject to

e1:  exp(x1) + 2 * exp(x2) + 2 * exp(x3) + exp(x6) + exp(x10) = 2;
e2:  exp(x4) + 2 * exp(x5) + exp(x6) + exp(x7) = 1;
e3:  exp(x3) + exp(x7) + exp(x8) + 2 * exp(x9) + exp(x10) = 1;
