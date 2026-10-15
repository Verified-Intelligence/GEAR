# DNLP written by GAMS Convert at 03/06/26 16:51:14
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

var x1 >= -1, <= 1;
var x2 >= -1, <= 1;
var x3 >= -1, <= 1;
var x4 >= -1, <= 1;
var x5 >= -1, <= 1;
var x6 >= -1, <= 1;
var x7 >= -1, <= 1;
var x8 >= -1, <= 1;
var x9 >= -1, <= 1;
var x10 >= -1, <= 1;

minimize obj: -exp(-0.5 * (x1 * x1 + x2 * x2 + x3 * x3 + x4 * x4 + x5 * x5 + x6
    * x6 + x7 * x7 + x8 * x8 + x9 * x9 + x10 * x10));

