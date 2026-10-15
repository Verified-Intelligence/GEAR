# DNLP written by GAMS Convert at 03/06/26 16:51:16
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

var x1 >= -500, <= 500;
var x2 >= -500, <= 500;
var x3 >= -500, <= 500;
var x4 >= -500, <= 500;
var x5 >= -500, <= 500;
var x6 >= -500, <= 500;
var x7 >= -500, <= 500;
var x8 >= -500, <= 500;
var x9 >= -500, <= 500;
var x10 >= -500, <= 500;

minimize obj: -x1 * sin(sqrt(abs(x1))) - x2 * sin(sqrt(abs(x2))) - x3 * sin(
    sqrt(abs(x3))) - x4 * sin(sqrt(abs(x4))) - x5 * sin(sqrt(abs(x5))) - x6 *
    sin(sqrt(abs(x6))) - x7 * sin(sqrt(abs(x7))) - x8 * sin(sqrt(abs(x8))) - x9
    * sin(sqrt(abs(x9))) - x10 * sin(sqrt(abs(x10)));

