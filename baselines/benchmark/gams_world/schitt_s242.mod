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

var x1 := 2.5, >= 0, <= 10;
var x2 := 10, >= 0, <= 10;
var x3 := 10, >= 0, <= 10;

minimize obj: (exp(-0.1 * x1) - exp(-0.1 * x2) - 0.536957976864517 * x3)^2 + (
    exp(-0.2 * x1) - exp(-0.2 * x2) - 0.683395469841369 * x3)^2 + (exp(-0.3 *
    x1) - exp(-0.3 * x2) - 0.691031152313854 * x3)^2 + (exp(-0.4 * x1) - exp(
    -0.4 * x2) - 0.652004407146905 * x3)^2 + (exp(-0.5 * x1) - exp(-0.5 * x2)
    - 0.599792712713548 * x3)^2 + (exp(-0.6 * x1) - exp(-0.6 * x2) -
    0.54633288391736 * x3)^2 + (exp(-0.7 * x1) - exp(-0.7 * x2) -
    0.495673421825855 * x3)^2 + (exp(-0.8 * x1) - exp(-0.8 * x2) -
    0.448993501489319 * x3)^2 + (exp(-0.9 * x1) - exp(-0.9 * x2) -
    0.406446249936512 * x3)^2 + (exp(-x1) - exp(-x2) - 0.36783404124168 * x3)^2;

