# NLP written by GAMS Convert at 03/06/26 16:51:14
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

var x1 >= -30, <= 30;
var x2 >= -30, <= 30;
var x3 >= -30, <= 30;
var x4 >= -30, <= 30;
var x5 >= -30, <= 30;
var x6 >= -30, <= 30;
var x7 >= -30, <= 30;
var x8 >= -30, <= 30;
var x9 >= -30, <= 30;
var x10 >= -30, <= 30;

minimize obj: -20 * exp(-0.02 * sqrt(0.1 * (x1^2 + x2^2 + x3^2 + x4^2 + x5^2 +
    x6^2 + x7^2 + x8^2 + x9^2 + x10^2))) - exp(0.1 * (cos(6.28318530718 * x1)
    + cos(6.28318530718 * x2) + cos(6.28318530718 * x3) + cos(6.28318530718 *
    x4) + cos(6.28318530718 * x5) + cos(6.28318530718 * x6) + cos(6.28318530718
    * x7) + cos(6.28318530718 * x8) + cos(6.28318530718 * x9) + cos(
    6.28318530718 * x10))) + 22.718281828459;

