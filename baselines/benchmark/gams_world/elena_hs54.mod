# NLP written by GAMS Convert at 03/06/26 16:51:12
#
# Equation counts
#     Total        E        G        L        N        X        C        B
#         2        2        0        0        0        0        0        0
#
# Variable counts
#                  x        b        i      s1s      s2s       sc       si
#     Total     cont   binary  integer     sos1     sos2    scont     sint
#         7        7        0        0        0        0        0        0
# FX      0
#
# Nonzero counts
#     Total    const       NL
#         9        3        6
#
# Reformulation has removed 1 variable and 1 equation

var x1 := 2e+04, >= 0, <= 2e+04;
var x2 := 1, >= -1, <= 1;
var x3 := 5e+06, >= 0, <= 5e+06;
var x4 := 10, >= 0, <= 10;
var x5 := 1, >= -1, <= 1;
var x6 := 6e+07, >= 0, <= 6e+07;
var x7 >= 0, <= 1;

minimize obj: -exp(-0.5 * x7);

subject to

e1:  -2.12585034013605e-14 * (1.5625e-14 * (-1e+06 + x1)^2 + (-0.5 + 5e-05 * x1)
     * (-1 + x2) + (-1 + x2)^2) * (-2e+06 + x3)^2 - 0.0004 * (-10 + x4)^2 -
     400 * (-0.001 + x5)^2 - 4e-18 * (-1e+08 + x6)^2 + x7 = 0;
e2:  x1 + 4000 * x2 = 17600;
