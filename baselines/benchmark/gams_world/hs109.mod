# NLP written by GAMS Convert at 03/06/26 16:51:11
#
# Equation counts
#     Total        E        G        L        N        X        C        B
#        10        6        4        0        0        0        0        0
#
# Variable counts
#                  x        b        i      s1s      s2s       sc       si
#     Total     cont   binary  integer     sos1     sos2    scont     sint
#         9        9        0        0        0        0        0        0
# FX      0
#
# Nonzero counts
#     Total    const       NL
#        42        8       34
#
# Reformulation has removed 1 variable and 1 equation

var x1 >= 0, <= 1e+06;
var x2 >= 0, <= 1e+06;
var x3 >= -0.55, <= 0.55;
var x4 >= -0.55, <= 0.55;
var x5 := 196, >= 196, <= 252;
var x6 := 196, >= 196, <= 252;
var x7 := 196, >= 196, <= 252;
var x8 >= -400, <= 800;
var x9 >= -400, <= 800;

minimize obj: 1e-06 * x1^3 + 3 * x1 + 5.22074e-07 * x2^3 + 2 * x2;

subject to

e1:  -x3 + x4 >= -0.55;
e2:  x3 - x4 >= -0.55;
e3:  -x1^2 - x8^2 >= -2.25e+06;
e4:  -x2^2 - x9^2 >= -2.25e+06;
e5:  x5 * x6 * sin(-0.25 - x3) + x5 * x7 * sin(-0.25 - x4) + 0.494807918509046
     * x5^2 - 50.176 * x1 = -20070.4;
e6:  x5 * x6 * sin(-0.25 + x3) + x6 * x7 * sin(-0.25 + x3 - x4) +
     0.494807918509046 * x6^2 - 50.176 * x2 = -20070.4;
e7:  x5 * x7 * sin(-0.25 + x4) + x6 * x7 * sin(-0.25 - x3 + x4) +
     0.494807918509046 * x7^2 = -44244.143104;
e8:  x5 * x6 * cos(-0.25 - x3) + x5 * x7 * cos(-0.25 - x4) - 1.93782484342129
     * x5^2 + 0.0377975808 * x5^2 + 50.176 * x8 = 10035.2;
e9:  x5 * x6 * cos(-0.25 + x3) + x6 * x7 * cos(-0.25 + x3 - x4) -
     1.93782484342129 * x6^2 + 0.0377975808 * x6^2 + 50.176 * x9 = 10035.2;
e10:  x5 * x7 * cos(-0.25 + x4) + x6 * x7 * cos(-0.25 - x3 + x4) -
      1.93782484342129 * x7^2 + 0.0377975808 * x7^2 = -1150.937088;
