# NLP written by GAMS Convert at 03/06/26 16:51:15
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

var x1 >= -10, <= 10;
var x2 >= -10, <= 10;
var x3 >= -10, <= 10;

minimize obj: 1.04719755119667 * ((0.25 + 0.25 * x1)^2 * (sin(0.7853981633975 *
    x2)^2 + 1) + (0.25 + 0.25 * x2)^2 * (sin(0.7853981633975 * x3)^2 + 1) + 10
    * sin(3.9269908169875 + 0.7853981633975 * x1)^2 + (0.25 + 0.25 * x3)^2);

