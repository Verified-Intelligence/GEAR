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

var x1 >= -512, <= 512;
var x2 >= -512, <= 512;
var x3 >= -512, <= 512;
var x4 >= -512, <= 512;
var x5 >= -512, <= 512;
var x6 >= -512, <= 512;
var x7 >= -512, <= 512;
var x8 >= -512, <= 512;
var x9 >= -512, <= 512;
var x10 >= -512, <= 512;

minimize obj: x1 * x1 - 10 * cos(6.28318530718 * x1) + x2 * x2 - 10 * cos(
    6.28318530718 * x2) + x3 * x3 - 10 * cos(6.28318530718 * x3) + x4 * x4 - 10
    * cos(6.28318530718 * x4) + x5 * x5 - 10 * cos(6.28318530718 * x5) + x6 *
    x6 - 10 * cos(6.28318530718 * x6) + x7 * x7 - 10 * cos(6.28318530718 * x7)
    + x8 * x8 - 10 * cos(6.28318530718 * x8) + x9 * x9 - 10 * cos(
    6.28318530718 * x9) + x10 * x10 - 10 * cos(6.28318530718 * x10) + 100;

