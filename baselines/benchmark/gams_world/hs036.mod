# NLP written by GAMS Convert at 03/06/26 16:51:09
#
# Equation counts
#     Total        E        G        L        N        X        C        B
#         1        0        0        1        0        0        0        0
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

var x1 := 10, >= 0, <= 20;
var x2 := 10, >= 0, <= 11;
var x3 := 10, >= 0, <= 42;

minimize obj: -x1 * x2 * x3;

subject to

e1:  x1 + 2 * x2 + 2 * x3 <= 72;
