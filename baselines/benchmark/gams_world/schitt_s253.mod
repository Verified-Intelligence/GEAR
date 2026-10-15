# NLP written by GAMS Convert at 03/06/26 16:51:13
#
# Equation counts
#     Total        E        G        L        N        X        C        B
#         1        0        1        0        0        0        0        0
#
# Variable counts
#                  x        b        i      s1s      s2s       sc       si
#     Total     cont   binary  integer     sos1     sos2    scont     sint
#         3        3        0        0        0        0        0        0
# FX      0
#
# Nonzero counts
#     Total    const       NL
#         2        2        0
#
# Reformulation has removed 1 variable and 1 equation

var x1 >= 0, <= 42;
var x2 := 2, >= 0, <= 42;
var x3 >= 0, <= 42;

minimize obj: sqrt((-x1)^2 + (-x2)^2 + (-x3)^2) + sqrt((10 - x1)^2 + (-x2)^2 +
    (-x3)^2) + sqrt((10 - x1)^2 + (10 - x2)^2 + (-x3)^2) + sqrt((-x1)^2 + (10
    - x2)^2 + (-x3)^2) + sqrt((-x1)^2 + (-x2)^2 + (10 - x3)^2) + sqrt((10 - x1)
    ^2 + (-x2)^2 + (10 - x3)^2) + sqrt((10 - x1)^2 + (10 - x2)^2 + (10 - x3)^2)
    + sqrt((-x1)^2 + (10 - x2)^2 + (10 - x3)^2);

subject to

e1:  -3 * x1 - 3 * x3 >= -30;
