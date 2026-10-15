# NLP written by GAMS Convert at 03/06/26 16:51:13
#
# Equation counts
#     Total        E        G        L        N        X        C        B
#         0        0        0        0        0        0        0        0
#
# Variable counts
#                  x        b        i      s1s      s2s       sc       si
#     Total     cont   binary  integer     sos1     sos2    scont     sint
#         5        5        0        0        0        0        0        0
# FX      0
#
# Nonzero counts
#     Total    const       NL
#         0        0        0
#
# Reformulation has removed 1 variable and 1 equation

var x1 := 0.5, >= -0.5, <= 0.5;
var x2 := 1.5, >= 1.5, <= 2.5;
var x3 := -1, >= -2, <= -1;
var x4 := 0.01, >= 0.001, <= 0.1;
var x5 := 0.02, >= 0.001, <= 0.1;

minimize obj: (0.844 - x1 - x2 - x3)^2 + (-x2 * exp(-10 * x4) - x3 * exp(-10 *
    x5) + 0.908 - x1)^2 + (-x2 * exp(-20 * x4) - x3 * exp(-20 * x5) + 0.932 -
    x1)^2 + (-x2 * exp(-30 * x4) - x3 * exp(-30 * x5) + 0.936 - x1)^2 + (-x2 *
    exp(-40 * x4) - x3 * exp(-40 * x5) + 0.925 - x1)^2 + (-x2 * exp(-50 * x4)
    - x3 * exp(-50 * x5) + 0.908 - x1)^2 + (-x2 * exp(-60 * x4) - x3 * exp(-60
    * x5) + 0.881 - x1)^2 + (-x2 * exp(-70 * x4) - x3 * exp(-70 * x5) + 0.85
    - x1)^2 + (-x2 * exp(-80 * x4) - x3 * exp(-80 * x5) + 0.818 - x1)^2 + (-x2
    * exp(-90 * x4) - x3 * exp(-90 * x5) + 0.784 - x1)^2 + (-x2 * exp(-100 *
    x4) - x3 * exp(-100 * x5) + 0.751 - x1)^2 + (-x2 * exp(-110 * x4) - x3 *
    exp(-110 * x5) + 0.718 - x1)^2 + (-x2 * exp(-120 * x4) - x3 * exp(-120 * x5)
    + 0.685 - x1)^2 + (-x2 * exp(-130 * x4) - x3 * exp(-130 * x5) + 0.658 - x1)
    ^2 + (-x2 * exp(-140 * x4) - x3 * exp(-140 * x5) + 0.628 - x1)^2 + (-x2 *
    exp(-150 * x4) - x3 * exp(-150 * x5) + 0.603 - x1)^2 + (-x2 * exp(-160 * x4)
    - x3 * exp(-160 * x5) + 0.58 - x1)^2 + (-x2 * exp(-170 * x4) - x3 * exp(
    -170 * x5) + 0.558 - x1)^2 + (-x2 * exp(-180 * x4) - x3 * exp(-180 * x5) +
    0.538 - x1)^2 + (-x2 * exp(-190 * x4) - x3 * exp(-190 * x5) + 0.522 - x1)^2
    + (-x2 * exp(-200 * x4) - x3 * exp(-200 * x5) + 0.506 - x1)^2 + (-x2 *
    exp(-210 * x4) - x3 * exp(-210 * x5) + 0.49 - x1)^2 + (-x2 * exp(-220 * x4)
    - x3 * exp(-220 * x5) + 0.478 - x1)^2 + (-x2 * exp(-230 * x4) - x3 * exp(
    -230 * x5) + 0.467 - x1)^2 + (-x2 * exp(-240 * x4) - x3 * exp(-240 * x5) +
    0.457 - x1)^2 + (-x2 * exp(-250 * x4) - x3 * exp(-250 * x5) + 0.448 - x1)^2
    + (-x2 * exp(-260 * x4) - x3 * exp(-260 * x5) + 0.438 - x1)^2 + (-x2 *
    exp(-270 * x4) - x3 * exp(-270 * x5) + 0.431 - x1)^2 + (-x2 * exp(-280 * x4)
    - x3 * exp(-280 * x5) + 0.424 - x1)^2 + (-x2 * exp(-290 * x4) - x3 * exp(
    -290 * x5) + 0.42 - x1)^2 + (-x2 * exp(-300 * x4) - x3 * exp(-300 * x5) +
    0.414 - x1)^2 + (-x2 * exp(-310 * x4) - x3 * exp(-310 * x5) + 0.411 - x1)^2
    + (-x2 * exp(-320 * x4) - x3 * exp(-320 * x5) + 0.406 - x1)^2;

