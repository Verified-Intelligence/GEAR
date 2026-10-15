# NLP written by GAMS Convert at 03/06/26 16:50:35
#
# Equation counts
#     Total        E        G        L        N        X        C        B
#         2        0        0        2        0        0        0        0
#
# Variable counts
#                  x        b        i      s1s      s2s       sc       si
#     Total     cont   binary  integer     sos1     sos2    scont     sint
#         4        4        0        0        0        0        0        0
# FX      0
#
# Nonzero counts
#     Total    const       NL
#         8        0        8
#
# Reformulation has removed 1 variable and 1 equation

var x1 := 200, >= 100, <= 4e+05;
var x2 := 200, >= 100, <= 3e+05;
var x3 := 200, >= 100, <= 2e+05;
var x4 := 200, >= 100, <= 1e+05;

minimize obj: x1 + x2 + x3 + x4;

subject to

e1:  4 / x1 + 2.25 / x2 + 1 / x3 + 0.25 / x4 <= 0.0401;
e2:  0.16 / x1 + 0.36 / x2 + 0.64 / x3 + 0.64 / x4 <= 0.010085;
