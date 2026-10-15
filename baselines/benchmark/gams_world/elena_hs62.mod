# NLP written by GAMS Convert at 03/06/26 16:51:12
#
# Equation counts
#     Total        E        G        L        N        X        C        B
#         1        1        0        0        0        0        0        0
#
# Variable counts
#                  x        b        i      s1s      s2s       sc       si
#     Total     cont   binary  integer     sos1     sos2    scont     sint
#         3        3        0        0        0        0        0        0
# FX      0
#
# Nonzero counts
#     Total    const       NL
#         3        3        0
#
# Reformulation has removed 1 variable and 1 equation

var x1 := 0.7, >= 0, <= 1;
var x2 := 0.2, >= 0, <= 1;
var x3 := 0.1, >= 0, <= 1;

minimize obj: -32.174 * (255 * log((0.03 + x1 + x2 + x3) / (0.03 + 0.09 * x1 +
    x2 + x3)) + 280 * log((0.03 + x2 + x3) / (0.03 + 0.07 * x2 + x3)) + 290 *
    log((0.03 + x3) / (0.03 + 0.13 * x3)));

subject to

e1:  x1 + x2 + x3 = 1;
