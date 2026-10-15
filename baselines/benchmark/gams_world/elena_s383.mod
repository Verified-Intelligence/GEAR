# NLP written by GAMS Convert at 03/06/26 16:51:12
#
# Equation counts
#     Total        E        G        L        N        X        C        B
#         1        1        0        0        0        0        0        0
#
# Variable counts
#                  x        b        i      s1s      s2s       sc       si
#     Total     cont   binary  integer     sos1     sos2    scont     sint
#        14       14        0        0        0        0        0        0
# FX      0
#
# Nonzero counts
#     Total    const       NL
#        14       14        0
#
# Reformulation has removed 1 variable and 1 equation

var x1 := 0.01, >= 0, <= 0.04;
var x2 := 0.01, >= 0, <= 0.04;
var x3 := 0.01, >= 0, <= 0.04;
var x4 := 0.01, >= 0, <= 0.04;
var x5 := 0.01, >= 0, <= 0.04;
var x6 := 0.01, >= 0, <= 0.03;
var x7 := 0.01, >= 0, <= 0.03;
var x8 := 0.01, >= 0, <= 0.03;
var x9 := 0.01, >= 0, <= 0.03;
var x10 := 0.01, >= 0, <= 0.03;
var x11 := 0.01, >= 0, <= 0.03;
var x12 := 0.01, >= 0, <= 0.03;
var x13 := 0.01, >= 0, <= 0.03;
var x14 := 0.01, >= 0, <= 0.03;

minimize obj: 12842.275 / x1 + 634.25 / x2 + 634.25 / x3 + 634.125 / x4 + 1268
    / x5 + 633.875 / x6 + 633.75 / x7 + 1267 / x8 + 760.05 / x9 + 633.25 / x10
    + 1266.25 / x11 + 632.875 / x12 + 394.46 / x13 + 940.838 / x14;

subject to

e1:  5.47934 * x1 + 0.83234 * x2 + 0.94749 * x3 + 1.11082 * x4 + 2.64824 * x5
     + 1.55868 * x6 + 1.73215 * x7 + 3.90896 * x8 + 2.74284 * x9 + 2.60541 *
     x10 + 5.96184 * x11 + 3.29522 * x12 + 1.83517 * x13 + 2.81372 * x14 = 1;
