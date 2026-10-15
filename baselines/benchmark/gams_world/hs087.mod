# DNLP written by GAMS Convert at 03/06/26 16:51:10
#
# Equation counts
#     Total        E        G        L        N        X        C        B
#         4        4        0        0        0        0        0        0
#
# Variable counts
#                  x        b        i      s1s      s2s       sc       si
#     Total     cont   binary  integer     sos1     sos2    scont     sint
#         6        6        0        0        0        0        0        0
# FX      0
#
# Nonzero counts
#     Total    const       NL
#        15        3       12
#
# Reformulation has removed 1 variable and 1 equation

var x1 := 390, >= 0, <= 400;
var x2 := 1000, >= 0, <= 1000;
var x3 := 419.5, >= 340, <= 420;
var x4 := 340.5, >= 340, <= 420;
var x5 := 198.175, >= -1000, <= 1000;
var x6 := 0.5, >= 0, <= 0.5236;

minimize obj: x1 * (30 * (abs(-300 + x1) - x1) + 9000) / (abs(-300 + x1) - x1
    + 300.0000000001) + x1 * (31 * (abs(-300 + x1) + x1) - 9300) / (abs(-300
    + x1) + x1 - 299.9999999999) + x2 * (28 * (abs(-100 + x2) - x2) + 2800) /
    (abs(-100 + x2) - x2 + 100.0000000001) + x2 * (29 * (abs(-100 + x2) + x2)
    - 2900) / (abs(-100 + x2) + x2 - 99.9999999999) * (abs(-200 + x2) - x2 +
    200) / (abs(-200 + x2) - x2 + 200.0000000001) + x2 * (30 * (abs(-200 + x2)
    + x2) - 6000) / (abs(-200 + x2) + x2 - 199.9999999999);

subject to

e1:  -0.000656500561892293 * x3^2 + 0.007629045301271 * x3 * x4 * cos(1.48477
     - x6) + x1 = 300;
e2:  -0.000656500561892293 * x4^2 + 0.007629045301271 * x3 * x4 * cos(1.48477
     + x6) + x2 = 0;
e3:  -0.00689584082973523 * x4^2 + 0.007629045301271 * x3 * x4 * sin(1.48477 +
     x6) + x5 = 0;
e4:  -0.00689584082973523 * x3^2 + 0.007629045301271 * x3 * x4 * sin(1.48477 -
     x6) = 200;
