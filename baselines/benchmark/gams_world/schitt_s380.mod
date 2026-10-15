# NLP written by GAMS Convert at 03/06/26 16:51:13
#
# Equation counts
#     Total        E        G        L        N        X        C        B
#         3        0        3        0        0        0        0        0
#
# Variable counts
#                  x        b        i      s1s      s2s       sc       si
#     Total     cont   binary  integer     sos1     sos2    scont     sint
#        12       12        0        0        0        0        0        0
# FX      0
#
# Nonzero counts
#     Total    const       NL
#        23        9       14
#
# Reformulation has removed 1 variable and 1 equation

var x1 := 4, >= 0.1, <= 100;
var x2 := 4, >= 0.1, <= 100;
var x3 := 4, >= 0.1, <= 100;
var x4 := 4, >= 0.1, <= 100;
var x5 := 4, >= 0.1, <= 100;
var x6 := 4, >= 0.1, <= 100;
var x7 := 4, >= 0.1, <= 100;
var x8 := 4, >= 0.1, <= 100;
var x9 := 4, >= 0.1, <= 100;
var x10 := 4, >= 0.1, <= 100;
var x11 := 4, >= 0.1, <= 100;
var x12 := 4, >= 0.1, <= 100;

minimize obj: 1e+05 * exp(-0.00133172 * log(x1)) * exp(-0.002270927 * log(x2))
    * exp(-0.00248546 * log(x3)) * exp(-4.67 * log(x4)) * exp(-4.671973 * log(
    x5)) * exp(-0.00814 * log(x6)) * exp(-0.008092 * log(x7)) * exp(-0.005 *
    log(x8)) * exp(-0.000909 * log(x9)) * exp(-0.00088 * log(x10)) * exp(-
    0.00119 * log(x11));

subject to

e1:  -0.0066940803 * x4 * x5 - 0.05367373 * x1 - 0.021863746 * x2 - 0.097733533
     * x3 >= -1;
e2:  -1e-10 * x4 * x12 - 1e-08 * x5 / x12 - 0.01 * x6 / x12 - 0.0001 * x7 * x12
     - 0.10898645 * x4 * x5 - 0.00016108052 * x2 * x5 / x12 - 1e-05 * x2 -
     1e-23 * x2 * x4 * x5 - 1.9304541e-06 * x2 * x5 / (x4 * x12^2) - 0.001 *
     x10 / x12 - 1e-06 * x1 - 1e-06 * x3 >= -1;
e3:  -0.10898645 * x4 * x5 - 1e-09 * x4 - 1e-09 * x5 - 1.6108052e-05 * x2 * x5
     - 1e-05 * x2 - 1e-23 * x2 * x4 * x5 - 1.9304541e-08 * x2 * x5 / x4 -
     0.00011184059 * x1 * x9 - 1e-06 * x1 - 1e-05 * x9 - 1e-06 * x3 - 0.001 *
     x6 - 0.001 * x8 - 0.0001 * x11 >= -1;
