# NLP written by GAMS Convert at 03/06/26 16:51:13
#
# Equation counts
#     Total        E        G        L        N        X        C        B
#         0        0        0        0        0        0        0        0
#
# Variable counts
#                  x        b        i      s1s      s2s       sc       si
#     Total     cont   binary  integer     sos1     sos2    scont     sint
#         8        8        0        0        0        0        0        0
# FX      0
#
# Nonzero counts
#     Total    const       NL
#         0        0        0
#
# Reformulation has removed 1 variable and 1 equation

var x1 := 1, >= 0, <= 1;
var x2 := 1, >= 0, <= 1;
var x3 := 1, >= 0, <= 1;
var x4 := 1, >= 0, <= 1;
var x5 := 1, >= 0, <= 1;
var x6 := 1, >= 0, <= 1;
var x7 := 1, >= 0, <= 1;
var x8 := 1, >= 0, <= 1;

minimize obj: (x1^3 + x2^3 + x3^3 + x4^3 + x5^3 + x6^3 + x7^3 + x8^3)^2 - (x1^2
    + x2^2 + x3^2 + x4^2 + x5^2 + x6^2 + x7^2 + x8^2) * (x1^4 + x2^4 + x3^4 +
    x4^4 + x5^4 + x6^4 + x7^4 + x8^4);

