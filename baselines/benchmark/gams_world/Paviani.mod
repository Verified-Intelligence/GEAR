# NLP written by GAMS Convert at 03/06/26 16:51:15
#
# Equation counts
#     Total        E        G        L        N        X        C        B
#         0        0        0        0        0        0        0        0
#
# Variable counts
#                  x        b        i      s1s      s2s       sc       si
#     Total     cont   binary  integer     sos1     sos2    scont     sint
#        10       10        0        0        0        0        0        0
# FX      0
#
# Nonzero counts
#     Total    const       NL
#         0        0        0
#
# Reformulation has removed 1 variable and 1 equation

var x1 := 2.000001, >= 2.000001, <= 9.999999;
var x2 := 2.000001, >= 2.000001, <= 9.999999;
var x3 := 2.000001, >= 2.000001, <= 9.999999;
var x4 := 2.000001, >= 2.000001, <= 9.999999;
var x5 := 2.000001, >= 2.000001, <= 9.999999;
var x6 := 2.000001, >= 2.000001, <= 9.999999;
var x7 := 2.000001, >= 2.000001, <= 9.999999;
var x8 := 2.000001, >= 2.000001, <= 9.999999;
var x9 := 2.000001, >= 2.000001, <= 9.999999;
var x10 := 2.000001, >= 2.000001, <= 9.999999;

minimize obj: log(-2 + x1)^2 + log(10 - x1)^2 + log(-2 + x2)^2 + log(10 - x2)^2
    + log(-2 + x3)^2 + log(10 - x3)^2 + log(-2 + x4)^2 + log(10 - x4)^2 + log(
    -2 + x5)^2 + log(10 - x5)^2 + log(-2 + x6)^2 + log(10 - x6)^2 + log(-2 + x7)
    ^2 + log(10 - x7)^2 + log(-2 + x8)^2 + log(10 - x8)^2 + log(-2 + x9)^2 +
    log(10 - x9)^2 + log(-2 + x10)^2 + log(10 - x10)^2 - exp(0.2 * log(x1 * x2
    * x3 * x4 * x5 * x6 * x7 * x8 * x9 * x10));

