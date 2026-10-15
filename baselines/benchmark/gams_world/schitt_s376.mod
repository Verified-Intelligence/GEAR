# NLP written by GAMS Convert at 03/06/26 16:51:13
#
# Equation counts
#     Total        E        G        L        N        X        C        B
#        15        1       14        0        0        0        0        0
#
# Variable counts
#                  x        b        i      s1s      s2s       sc       si
#     Total     cont   binary  integer     sos1     sos2    scont     sint
#        10       10        0        0        0        0        0        0
# FX      0
#
# Nonzero counts
#     Total    const       NL
#        50       15       35
#
# Reformulation has removed 1 variable and 1 equation

var x1 := 10, >= 0, <= 10;
var x2 := 0.005, >= 0, <= 0.1;
var x3 := 0.0081, >= 5e-05, <= 0.0081;
var x4 := 100, >= 10, <= 1000;
var x5 := 0.0017, >= 5e-05, <= 0.0017;
var x6 := 0.0013, >= 5e-05, <= 0.0013;
var x7 := 0.0027, >= 5e-05, <= 0.0027;
var x8 := 0.002, >= 5e-05, <= 0.002;
var x9 := 0.15, >= 5e-05, <= 1;
var x10 := 0.105, >= 5e-05, <= 1;

minimize obj: -(-1200 + 3000 * x1 + 2.8e+05 * x2) / (0.002 + x1 + 60 * x2);

subject to

e1:  -0.75 / (x3 * x4) + x1 >= 0;
e2:  -x9 / (x5 * x4) + x1 >= 0;
e3:  -x10 / (x6 * x4) - 10 / x4 + x1 >= 0;
e4:  -0.19 / (x4 * x7) - 10 / x4 + x1 >= 0;
e5:  -0.125 / (x4 * x8) + x1 >= 0;
e6:  -0.00131 * exp(0.666 * log(x5)) * x9 * exp(1.5 * log(x4)) + 1e+04 * x2
     >= 0;
e7:  -0.001038 * exp(1.6 * log(x6)) * x10 * x4^3 + 1e+04 * x2 >= 0;
e8:  -0.000223 * exp(0.666 * log(x7)) * exp(1.5 * log(x4)) + 1e+04 * x2 >= 0;
e9:  -7.6e-05 * exp(3.55 * log(x8)) * exp(5.66 * log(x4)) + 1e+04 * x2 >= 0;
e10:  -0.000698 * exp(1.2 * log(x3)) * x4^2 + 1e+04 * x2 >= 0;
e11:  -5e-05 * exp(1.6 * log(x3)) * x4^3 + 1e+04 * x2 >= 0;
e12:  -6.54e-06 * exp(2.42 * log(x3)) * exp(4.17 * log(x4)) + 1e+04 * x2 >= 0;
e13:  -0.000257 * exp(0.666 * log(x3)) * exp(1.5 * log(x4)) + 1e+04 * x2 >= 0;
e14:  -2.003 * x4 * x5 - 1.885 * x6 * x4 - 0.184 * x8 * x4 - 2 * exp(0.803 *
      log(x3)) * x4 >= -30;
e15:  x9 + x10 = 0.255;
