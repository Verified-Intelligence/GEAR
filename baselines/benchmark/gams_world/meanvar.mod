# NLP written by GAMS Convert at 03/06/26 16:50:35
#
# Equation counts
#     Total        E        G        L        N        X        C        B
#         2        2        0        0        0        0        0        0
#
# Variable counts
#                  x        b        i      s1s      s2s       sc       si
#     Total     cont   binary  integer     sos1     sos2    scont     sint
#         8        8        0        0        0        0        0        0
# FX      1
#
# Nonzero counts
#     Total    const       NL
#        15       15        0
#
# Reformulation has removed 1 variable and 1 equation

var x1 := 0.115, >= 0.115, <= 0.115;
var x2 >= 0, <= 1;
var x3 >= 0, <= 1;
var x4 >= 0, <= 1;
var x5 >= 0, <= 1;
var x6 >= 0, <= 1;
var x7 >= 0, <= 1;
var x8 >= 0, <= 1;

minimize obj: 0.5 * (42.18 * x2 * x2 + 20.18 * x2 * x3 + 10.88 * x2 * x4 + 5.3
    * x2 * x5 + 12.32 * x2 * x6 + 23.84 * x2 * x7 + 17.41 * x2 * x8 + 20.18 *
    x3 * x2 + 70.89 * x3 * x3 + 21.58 * x3 * x4 + 15.41 * x3 * x5 + 23.24 * x3
    * x6 + 23.8 * x3 * x7 + 12.62 * x3 * x8 + 10.88 * x4 * x2 + 21.58 * x4 *
    x3 + 25.51 * x4 * x4 + 9.6 * x4 * x5 + 22.63 * x4 * x6 + 13.22 * x4 * x7 +
    4.7 * x4 * x8 + 5.3 * x5 * x2 + 15.41 * x5 * x3 + 9.6 * x5 * x4 + 22.33 *
    x5 * x5 + 10.32 * x5 * x6 + 10.46 * x5 * x7 + x5 * x8 + 12.32 * x6 * x2 +
    23.24 * x6 * x3 + 22.63 * x6 * x4 + 10.32 * x6 * x5 + 30.01 * x6 * x6 +
    16.36 * x6 * x7 + 7.2 * x6 * x8 + 23.84 * x7 * x2 + 23.8 * x7 * x3 + 13.22
    * x7 * x4 + 10.46 * x7 * x5 + 16.36 * x7 * x6 + 42.23 * x7 * x7 + 9.9 * x7
    * x8 + 17.41 * x8 * x2 + 12.62 * x8 * x3 + 4.7 * x8 * x4 + x8 * x5 + 7.2 *
    x8 * x6 + 9.9 * x8 * x7 + 16.42 * x8 * x8);

subject to

e1:  x1 - 0.1287 * x2 - 0.1096 * x3 - 0.0501 * x4 - 0.1524 * x5 - 0.0763 * x6
     - 0.1854 * x7 - 0.062 * x8 = 0;
e2:  x2 + x3 + x4 + x5 + x6 + x7 + x8 = 1;
