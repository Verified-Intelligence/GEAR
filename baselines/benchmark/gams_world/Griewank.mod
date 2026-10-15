# NLP written by GAMS Convert at 03/06/26 16:51:14
#
# Equation counts
#     Total        E        G        L        N        X        C        B
#         0        0        0        0        0        0        0        0
#
# Variable counts
#                  x        b        i      s1s      s2s       sc       si
#     Total     cont   binary  integer     sos1     sos2    scont     sint
#        10       10        0        0        0        0        0        0
# FX      0
#
# Nonzero counts
#     Total    const       NL
#         0        0        0
#
# Reformulation has removed 1 variable and 1 equation

var x1 >= -500, <= 500;
var x2 >= -500, <= 500;
var x3 >= -500, <= 500;
var x4 >= -500, <= 500;
var x5 >= -500, <= 500;
var x6 >= -500, <= 500;
var x7 >= -500, <= 500;
var x8 >= -500, <= 500;
var x9 >= -500, <= 500;
var x10 >= -500, <= 500;

minimize obj: 0.00025 * (x1 * x1 + x2 * x2 + x3 * x3 + x4 * x4 + x5 * x5 + x6
    * x6 + x7 * x7 + x8 * x8 + x9 * x9 + x10 * x10) - cos(x1) * cos(
    0.707106781186547 * x2) * cos(0.577350269189626 * x3) * cos(0.5 * x4) *
    cos(0.447213595499958 * x5) * cos(0.408248290463863 * x6) * cos(
    0.377964473009227 * x7) * cos(0.353553390593274 * x8) * cos(
    0.333333333333333 * x9) * cos(0.316227766016838 * x10) + 1;

