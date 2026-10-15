# NLP written by GAMS Convert at 03/06/26 16:51:10
#
# Equation counts
#     Total        E        G        L        N        X        C        B
#         6        0        5        1        0        0        0        0
#
# Variable counts
#                  x        b        i      s1s      s2s       sc       si
#     Total     cont   binary  integer     sos1     sos2    scont     sint
#         7        7        0        0        0        0        0        0
# FX      0
#
# Nonzero counts
#     Total    const       NL
#        38        0       38
#
# Reformulation has removed 1 variable and 1 equation

var x1 := 6, >= 0.1, <= 10;
var x2 := 6, >= 0.1, <= 10;
var x3 := 6, >= 0.1, <= 10;
var x4 := 6, >= 0.1, <= 10;
var x5 := 6, >= 0.1, <= 10;
var x6 := 6, >= 0.1, <= 10;
var x7 := 6, >= 0.01, <= 10;

minimize obj: 10 * x4^2 * x1 * exp(0.125 * log(x7)) / (x2 * x6^3) + 15 * x3 *
    x4 / (x1 * x2^2 * x5 * sqrt(x7)) + 20 * x2 * x6 / (x4 * x1^2 * x5^2) + 25
    * x1^2 * x2^2 * sqrt(x5) * x7 / (x3 * x6^2);

subject to

e1:  -0.5 * sqrt(x1) * x7 / (x3 * x6^2) - 0.7 * x1^3 * x2 * x6 * sqrt(x7) / x3^
     2 - 0.2 * exp(0.666666666666667 * log(x6)) * x3 * (x7^2)^2 / (x2 * sqrt(x4))
     >= -1;
e2:  -1.3 * x2 * x6 / (x3 * sqrt(x1) * x5) - 0.8 * x6^2 * x3 / (x4 * x5) - 3.1
     * sqrt(x2) * exp(0.333333333333333 * log(x6)) / (x1 * x4^2 * x5) >= -1;
e3:  -2 * x1 * x5 * exp(0.333333333333333 * log(x7)) / (x6 * exp(1.5 * log(x3)))
     - 0.1 * x2 * x5 / (x6 * sqrt(x3) * sqrt(x7)) - x2 * sqrt(x3) * x5 / x1 -
     0.65 * x3 * x5 * x7 / (x6 * x2^2) >= -1;
e4:  -0.2 * sqrt(x5) * x2 * exp(0.333333333333333 * log(x7)) / (x4 * x1^2) -
     0.3 * sqrt(x1) * x2^2 * x3 * exp(0.333333333333333 * log(x4)) * (x7^2)^2
     / exp(0.666666666666667 * log(x5)) - 0.4 * x3 * x5 * exp(0.75 * log(x7))
     / (x1^3 * x2^2) - 0.5 * sqrt(x7) * x4 / x3^2 >= -1;
e5:  10 * x4^2 * x1 * exp(0.125 * log(x7)) / (x2 * x6^3) + 15 * x3 * x4 / (x1
     * x2^2 * x5 * sqrt(x7)) + 20 * x2 * x6 / (x4 * x1^2 * x5^2) + 25 * x1^2
     * x2^2 * sqrt(x5) * x7 / (x3 * x6^2) >= 100;
e6:  10 * x4^2 * x1 * exp(log(0.125 * x7)) / (x2 * x6^3) + 15 * x3 * x4 / (x1
     * x2^2 * x5 * sqrt(x7)) + 20 * x2 * x6 / (x4 * x1^2 * x5^2) + 25 * x1^2
     * x2^2 * sqrt(x5) * x7 / (x3 * x6^2) <= 3000;
