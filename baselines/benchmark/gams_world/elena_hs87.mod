# NLP written by GAMS Convert at 03/06/26 16:51:12
#
# Equation counts
#     Total        E        G        L        N        X        C        B
#         6        6        0        0        0        0        0        0
#
# Variable counts
#                  x        b        i      s1s      s2s       sc       si
#     Total     cont   binary  integer     sos1     sos2    scont     sint
#        11       11        0        0        0        0        0        0
# FX      0
#
# Nonzero counts
#     Total    const       NL
#        22       10       12
#
# Reformulation has removed 1 variable and 1 equation

var x1 := 390, >= 0, <= 400;
var x2 := 1000, >= 0, <= 1000;
var x3 := 419.5, >= 340, <= 420;
var x4 := 340.5, >= 340, <= 420;
var x5 := 198.175, >= -1000, <= 1000;
var x6 := 0.5, >= 0, <= 0.5236;
var x7 >= 0, <= 300;
var x8 >= 0, <= 100;
var x9 >= 0, <= 100;
var x10 >= 0, <= 100;
var x11 >= 0, <= 800;

minimize obj: 30 * x7 + 31 * x8 + 28 * x9 + 29 * x10 + 30 * x11;

subject to

e1:  -0.000656500561892293 * x3^2 + 0.007629045301271 * x3 * x4 * cos(1.48477
     - x6) + x1 = 300;
e2:  -0.000656500561892293 * x4^2 + 0.007629045301271 * x3 * x4 * cos(1.48477
     + x6) + x2 = 0;
e3:  -0.00689584082973523 * x4^2 + 0.007629045301271 * x3 * x4 * sin(1.48477 +
     x6) + x5 = 0;
e4:  -0.00689584082973523 * x3^2 + 0.007629045301271 * x3 * x4 * sin(1.48477 -
     x6) = 200;
e5:  x1 - x7 - x8 = 0;
e6:  x2 - x9 - x10 - x11 = 0;
