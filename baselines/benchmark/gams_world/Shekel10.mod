# NLP written by GAMS Convert at 03/06/26 16:51:16
#
# Equation counts
#     Total        E        G        L        N        X        C        B
#         0        0        0        0        0        0        0        0
#
# Variable counts
#                  x        b        i      s1s      s2s       sc       si
#     Total     cont   binary  integer     sos1     sos2    scont     sint
#         4        4        0        0        0        0        0        0
# FX      0
#
# Nonzero counts
#     Total    const       NL
#         0        0        0
#
# Reformulation has removed 1 variable and 1 equation

var x1 >= 0, <= 10;
var x2 >= 0, <= 10;
var x3 >= 0, <= 10;
var x4 >= 0, <= 10;

minimize obj: -1 / ((-4 + x1)^2 + (-4 + x2)^2 + (-4 + x3)^2 + (-4 + x4)^2 + 0.1)
    - 1 / ((-1 + x1)^2 + (-1 + x2)^2 + (-1 + x3)^2 + (-1 + x4)^2 + 0.2) - 1 /
    ((-8 + x1)^2 + (-8 + x2)^2 + (-8 + x3)^2 + (-8 + x4)^2 + 0.2) - 1 / ((-6 +
    x1)^2 + (-6 + x2)^2 + (-6 + x3)^2 + (-6 + x4)^2 + 0.4) - 1 / ((-3 + x1)^2
    + (-7 + x2)^2 + (-3 + x3)^2 + (-7 + x4)^2 + 0.4) - 1 / ((-2 + x1)^2 + (-9
    + x2)^2 + (-2 + x3)^2 + (-9 + x4)^2 + 0.6) - 1 / ((-5 + x1)^2 + (-5 + x2)^
    2 + (-3 + x3)^2 + (-3 + x4)^2 + 0.3) - 1 / ((-8 + x1)^2 + (-1 + x2)^2 + (-8
    + x3)^2 + (-1 + x4)^2 + 0.7) - 1 / ((-6 + x1)^2 + (-2 + x2)^2 + (-6 + x3)^
    2 + (-2 + x4)^2 + 0.5) - 1 / ((-7 + x1)^2 + (-3.6 + x2)^2 + (-7 + x3)^2 + (
    -3.6 + x4)^2 + 0.5);

