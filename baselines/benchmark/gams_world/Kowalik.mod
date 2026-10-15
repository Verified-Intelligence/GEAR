# DNLP written by GAMS Convert at 03/06/26 16:51:15
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

var x1 >= 0, <= 0.42;
var x2 >= 0, <= 0.42;
var x3 >= 0, <= 0.42;
var x4 >= 0, <= 0.42;

minimize obj: (-x1 * (1 + 0.25 * x2) / (1 + 0.25 * x3 + 0.0625 * x4) + 0.1957)^
    2 + (-x1 * (1 + 0.5 * x2) / (1 + 0.5 * x3 + 0.25 * x4) + 0.1947)^2 + (-x1
    * (1 + x2) / (1 + x3 + x4) + 0.1735)^2 + (-x1 * (1 + 2 * x2) / (1 + 2 * x3
    + 4 * x4) + 0.16)^2 + (-x1 * (1 + 4 * x2) / (1 + 4 * x3 + 16 * x4) +
    0.0844)^2 + (-x1 * (1 + 6 * x2) / (1 + 6 * x3 + 36 * x4) + 0.0627)^2 + (-x1
    * (1 + 8 * x2) / (1 + 8 * x3 + 64 * x4) + 0.0456)^2 + (-x1 * (1 + 10 * x2)
    / (1 + 10 * x3 + 100 * x4) + 0.0342)^2 + (-x1 * (1 + 12 * x2) / (1 + 12 *
    x3 + 144 * x4) + 0.0323)^2 + (-x1 * (1 + 14 * x2) / (1 + 14 * x3 + 196 * x4)
    + 0.0235)^2 + (-x1 * (1 + 16 * x2) / (1 + 16 * x3 + 256 * x4) + 0.0246)^2;

