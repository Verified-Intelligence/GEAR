# NLP written by GAMS Convert at 03/06/26 16:50:56
#
# Equation counts
#     Total        E        G        L        N        X        C        B
#         6        0        5        1        0        0        0        0
#
# Variable counts
#                  x        b        i      s1s      s2s       sc       si
#     Total     cont   binary  integer     sos1     sos2    scont     sint
#         8        8        0        0        0        0        0        0
# FX      0
#
# Nonzero counts
#     Total    const       NL
#        21        3       18
#
# Reformulation has removed 1 variable and 1 equation

var x1 := 0.1, >= 0.1, <= 10;
var x2 := 3, >= 0.1, <= 10;
var x3 := 0.4, >= 0.1, <= 10;
var x4 := 0.2, >= 0.1, <= 10;
var x5 := 6, >= 0.1, <= 10;
var x6 := 6, >= 0.1, <= 10;
var x7 := 1, >= 0.1, <= 10;
var x8 := 0.5, >= 0.1, <= 10;

minimize obj: 0.4 * exp(0.67 * log(x1)) * exp(-0.67 * log(x7)) + 0.4 * exp(0.67
    * log(x2)) * exp(-0.67 * log(x8)) - x1 - x2 + 10;

subject to

e1:  -0.0588 * x5 * x7 - 0.1 * x1 >= -1;
e2:  -0.0588 * x6 * x8 - 0.1 * x1 - 0.1 * x2 >= -1;
e3:  -4 * x3 / x5 - 2 / (x5 * exp(0.71 * log(x3))) - (0.0588 * x7) / exp(1.3 *
     log(x3)) >= -1;
e4:  -4 * x4 / x6 - 2 / (x6 * exp(0.71 * log(x4))) - (0.0588 * x8) / exp(1.3 *
     log(x4)) >= -1;
e5:  0.4 * exp(0.67 * log(x1)) * exp(-0.67 * log(x7)) + 0.4 * exp(0.67 * log(x2))
     * exp(-0.67 * log(x8)) - x1 - x2 >= -9.9;
e6:  0.4 * exp(0.67 * log(x1)) * exp(-0.67 * log(x7)) + 0.4 * exp(0.67 * log(x2))
     * exp(-0.67 * log(x8)) - x1 - x2 <= -5.8;
