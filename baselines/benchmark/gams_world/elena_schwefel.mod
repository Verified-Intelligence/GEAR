# NLP written by GAMS Convert at 03/06/26 16:51:12
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

var x1 := 0.4, >= -0.5, <= 0.4;
var x2 := 0.4, >= -0.5, <= 0.4;
var x3 := 0.4, >= -0.5, <= 0.4;
var x4 := 0.4, >= -0.5, <= 0.4;
var x5 := 0.4, >= -0.5, <= 0.4;

minimize obj: x1^10 + x2^10 + x3^10 + x4^10 + x5^10;

