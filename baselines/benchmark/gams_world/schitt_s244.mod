# NLP written by GAMS Convert at 03/06/26 16:51:12
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

var x1 := 1, >= 0, <= 10;
var x2 := 2, >= 0, <= 10;
var x3 := 1, >= 0, <= 10;

minimize obj: (exp(-0.1 * x1) - x3 * exp(-0.1 * x2) + 0.934559787821252)^2 + (
    exp(-0.2 * x1) - x3 * exp(-0.2 * x2) - 0.142054336894918)^2 + (exp(-0.3 *
    x1) - x3 * exp(-0.3 * x2) - 0.491882878842398)^2;

