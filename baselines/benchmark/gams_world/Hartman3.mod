# NLP written by GAMS Convert at 03/06/26 16:51:14
#
# Equation counts
#     Total        E        G        L        N        X        C        B
#         0        0        0        0        0        0        0        0
#
# Variable counts
#                  x        b        i      s1s      s2s       sc       si
#     Total     cont   binary  integer     sos1     sos2    scont     sint
#         3        3        0        0        0        0        0        0
# FX      0
#
# Nonzero counts
#     Total    const       NL
#         0        0        0
#
# Reformulation has removed 1 variable and 1 equation

var x1 >= 0, <= 1;
var x2 >= 0, <= 1;
var x3 >= 0, <= 1;

minimize obj: -exp(-3 * (-0.3689 + x1)^2 - 10 * (-0.117 + x2)^2 - 30 * (-0.2673
    + x3)^2) - 1.2 * exp(-0.1 * (-0.4699 + x1)^2 - 10 * (-0.4387 + x2)^2 - 35
    * (-0.747 + x3)^2) - 3 * exp(-3 * (-0.1091 + x1)^2 - 10 * (-0.8732 + x2)^2
    - 30 * (-0.5547 + x3)^2) - 3.2 * exp(-0.1 * (-0.03815 + x1)^2 - 10 * (
    -0.5743 + x2)^2 - 35 * (-0.8828 + x3)^2);

