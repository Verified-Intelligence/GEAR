# NLP written by GAMS Convert at 03/06/26 16:50:56
#
# Equation counts
#     Total        E        G        L        N        X        C        B
#         6        0        0        6        0        0        0        0
#
# Variable counts
#                  x        b        i      s1s      s2s       sc       si
#     Total     cont   binary  integer     sos1     sos2    scont     sint
#         5        5        0        0        0        0        0        0
# FX      0
#
# Nonzero counts
#     Total    const       NL
#        26        0       26
#
# Reformulation has removed 1 variable and 1 equation

var x1 := 78, >= 78, <= 102;
var x2 := 33, >= 33, <= 45;
var x3 := 27, >= 27, <= 45;
var x4 := 27, >= 27, <= 45;
var x5 := 27, >= 27, <= 45;

minimize obj: 5.3578547 * x3^2 + 0.8356891 * x1 * x5 + 37.293239 * x1 -
    40792.141;

subject to

e1:  -0.0056858 * x2 * x5 - 0.0006262 * x1 * x4 + 0.0022053 * x3 * x5
     <= 85.334407;
e2:  0.0056858 * x2 * x5 + 0.0006262 * x1 * x4 - 0.0022053 * x3 * x5
     <= 6.665593;
e3:  -0.0071317 * x2 * x5 - 0.0029955 * x1 * x2 - 0.0021813 * x3^2 <= -9.48751;
e4:  0.0071317 * x2 * x5 + 0.0029955 * x1 * x2 + 0.0021813 * x3^2 <= 29.48751;
e5:  -0.0047026 * x3 * x5 - 0.0012547 * x1 * x3 - 0.0019085 * x3 * x4
     <= -10.699039;
e6:  0.0047026 * x3 * x5 + 0.0012547 * x1 * x3 + 0.0019085 * x3 * x4
     <= 15.699039;
