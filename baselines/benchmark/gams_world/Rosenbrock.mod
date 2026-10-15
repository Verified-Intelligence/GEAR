# DNLP written by GAMS Convert at 03/06/26 16:51:15
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

minimize obj: (100 * x1 * x1 - 100 * x2) * (x1 * x1 - x2) + (1 - x1) * (1 - x1)
    + (100 * x2 * x2 - 100 * x3) * (x2 * x2 - x3) + (1 - x2) * (1 - x2) + (100
    * x3 * x3 - 100 * x4) * (x3 * x3 - x4) + (1 - x3) * (1 - x3) + (100 * x4
    * x4 - 100 * x5) * (x4 * x4 - x5) + (1 - x4) * (1 - x4) + (100 * x5 * x5
    - 100 * x6) * (x5 * x5 - x6) + (1 - x5) * (1 - x5) + (100 * x6 * x6 - 100
    * x7) * (x6 * x6 - x7) + (1 - x6) * (1 - x6) + (100 * x7 * x7 - 100 * x8)
    * (x7 * x7 - x8) + (1 - x7) * (1 - x7) + (100 * x8 * x8 - 100 * x9) * (x8
    * x8 - x9) + (1 - x8) * (1 - x8) + (100 * x9 * x9 - 100 * x10) * (x9 * x9
    - x10) + (1 - x9) * (1 - x9);

