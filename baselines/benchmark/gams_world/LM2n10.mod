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

var x1 >= -5, <= 5;
var x2 >= -5, <= 5;
var x3 >= -5, <= 5;
var x4 >= -5, <= 5;
var x5 >= -5, <= 5;
var x6 >= -5, <= 5;
var x7 >= -5, <= 5;
var x8 >= -5, <= 5;
var x9 >= -5, <= 5;
var x10 >= -5, <= 5;

minimize obj: 0.1 * ((-1 + x1)^2 * (sin(9.42477796077 * x2)^2 + 1) + (-1 + x2)^
    2 * (sin(9.42477796077 * x3)^2 + 1) + (-1 + x3)^2 * (sin(9.42477796077 * x4)
    ^2 + 1) + (-1 + x4)^2 * (sin(9.42477796077 * x5)^2 + 1) + (-1 + x5)^2 * (
    sin(9.42477796077 * x6)^2 + 1) + (-1 + x6)^2 * (sin(9.42477796077 * x7)^2
    + 1) + (-1 + x7)^2 * (sin(9.42477796077 * x8)^2 + 1) + (-1 + x8)^2 * (sin(
    9.42477796077 * x9)^2 + 1) + (-1 + x9)^2 * (sin(9.42477796077 * x10)^2 + 1)
    + sin(9.42477796077 * x1)^2 + (-1 + x10)^2 * (sin(6.28318530718 * x10)^2
    + 1));

