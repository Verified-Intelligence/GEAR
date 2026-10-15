# NLP written by GAMS Convert at 03/06/26 16:51:13
#
# Equation counts
#     Total        E        G        L        N        X        C        B
#         3        1        2        0        0        0        0        0
#
# Variable counts
#                  x        b        i      s1s      s2s       sc       si
#     Total     cont   binary  integer     sos1     sos2    scont     sint
#        10       10        0        0        0        0        0        0
# FX      0
#
# Nonzero counts
#     Total    const       NL
#        24       24        0
#
# Reformulation has removed 1 variable and 1 equation

var x1 >= 0, <= 0.3;
var x2 >= 0, <= 0.3;
var x3 >= 0, <= 0.3;
var x4 >= 0, <= 0.3;
var x5 >= 0, <= 0.3;
var x6 >= 0, <= 0.3;
var x7 >= 0, <= 0.3;
var x8 >= 0, <= 0.3;
var x9 >= 0, <= 0.3;
var x10 >= 0, <= 0.3;

minimize obj: 0.1 * x1 * x1 + 19 * x2 * x2 - 2 * x2 * x3 + 4 * x2 * x4 + x2 *
    x5 + x2 * x6 + x2 * x7 + 0.5 * x2 * x8 + 10 * x2 * x9 + 5 * x2 * x10 - 2 *
    x3 * x2 + 28 * x3 * x3 + x3 * x4 + 2 * x3 * x5 + x3 * x6 + x3 * x7 - 2 * x3
    * x9 - x3 * x10 + 4 * x4 * x2 + x4 * x3 + 22 * x4 * x4 + x4 * x6 + 2 * x4
    * x7 + 3 * x4 * x9 + 4 * x4 * x10 + x5 * x2 + 2 * x5 * x3 + 4 * x5 * x5 -
    1.5 * x5 * x6 - 2 * x5 * x7 - x5 * x8 + x5 * x9 + x5 * x10 + x6 * x2 + x6
    * x3 + x6 * x4 - 1.5 * x6 * x5 + 3.5 * x6 * x6 + 2 * x6 * x7 + 0.5 * x6 *
    x8 + x6 * x9 + 1.5 * x6 * x10 + x7 * x2 + x7 * x3 + 2 * x7 * x4 - 2 * x7 *
    x5 + 2 * x7 * x6 + 5 * x7 * x7 + 0.5 * x7 * x8 + x7 * x9 + 2.5 * x7 * x10
    + 0.5 * x8 * x2 - x8 * x5 + 0.5 * x8 * x6 + 0.5 * x8 * x7 + x8 * x8 + 0.5
    * x8 * x9 + 0.5 * x8 * x10 + 10 * x9 * x2 - 2 * x9 * x3 + 3 * x9 * x4 + x9
    * x5 + x9 * x6 + x9 * x7 + 0.5 * x9 * x8 + 25 * x9 * x9 + 8 * x9 * x10 + 5
    * x10 * x2 - x10 * x3 + 4 * x10 * x4 + x10 * x5 + 1.5 * x10 * x6 + 2.5 *
    x10 * x7 + 0.5 * x10 * x8 + 8 * x10 * x9 + 16 * x10 * x10;

subject to

e1:  x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 = 1;
e2:  x1 + x2 + x3 + x4 >= 0.5;
e3:  5 * x1 + 17 * x2 + 26 * x3 + 12 * x4 + 8 * x5 + 9 * x6 + 7 * x7 + 6 * x8
     + 31 * x9 + 21 * x10 >= 25;
