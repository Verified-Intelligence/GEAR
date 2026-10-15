# NLP written by GAMS Convert at 03/06/26 16:50:41
#
# Equation counts
#     Total        E        G        L        N        X        C        B
#         0        0        0        0        0        0        0        0
#
# Variable counts
#                  x        b        i      s1s      s2s       sc       si
#     Total     cont   binary  integer     sos1     sos2    scont     sint
#         6        6        0        0        0        0        0        0
# FX      0
#
# Nonzero counts
#     Total    const       NL
#         0        0        0
#
# Reformulation has removed 1 variable and 1 equation

var x1 := 0.2, >= 0, <= 1;
var x2 := 0.2, >= 0, <= 1;
var x3 := 0.2, >= 0, <= 1;
var x4 := 0.2, >= 0, <= 1;
var x5 := 0.2, >= 0, <= 1;
var x6 := 0.2, >= 0, <= 1;

minimize obj: -exp(-10 * (-0.1312 + x1)^2 - 0.05 * (-0.1696 + x2)^2 - 17 * (
    -0.5569 + x3)^2 - 3.5 * (-0.0124 + x4)^2 - 1.7 * (-0.8283 + x5)^2 - 8 * (
    -0.5886 + x6)^2) - 1.2 * exp(-0.05 * (-0.2329 + x1)^2 - 10 * (-0.4135 + x2)
    ^2 - 17 * (-0.8307 + x3)^2 - 0.1 * (-0.3736 + x4)^2 - 8 * (-0.1004 + x5)^2
    - 14 * (-0.9991 + x6)^2) - 3 * exp(-3 * (-0.2348 + x1)^2 - 3.5 * (-0.1451
    + x2)^2 - 1.7 * (-0.3522 + x3)^2 - 10 * (-0.2883 + x4)^2 - 17 * (-0.3047
    + x5)^2 - 8 * (-0.665 + x6)^2) - 3.2 * exp(-17 * (-0.4047 + x1)^2 - 8 * (
    -0.8828 + x2)^2 - 0.05 * (-0.8732 + x3)^2 - 10 * (-0.5743 + x4)^2 - 0.1 * (
    -0.1091 + x5)^2 - 14 * (-0.0381 + x6)^2);

