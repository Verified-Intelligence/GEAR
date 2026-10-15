# NLP written by GAMS Convert at 03/06/26 16:50:41
#
# Equation counts
#     Total        E        G        L        N        X        C        B
#        13        0        1       12        0        0        0        0
#
# Variable counts
#                  x        b        i      s1s      s2s       sc       si
#     Total     cont   binary  integer     sos1     sos2    scont     sint
#         4        4        0        0        0        0        0        0
# FX      0
#
# Nonzero counts
#     Total    const       NL
#        28       28        0
#
# Reformulation has removed 1 variable and 1 equation

var x1 := 1, >= 0, <= 5;
var x2 := 5, >= 0, <= 5;
var x3 := 5, >= 0, <= 5;
var x4 := 1, >= 0, <= 5;

minimize obj: -x1 * x3 - x2 * x4;

subject to

e1:  -x1 - x2 <= -2.5;
e2:  -x1 - x3 <= -2.5;
e3:  -x1 - x4 <= -2.5;
e4:  -x2 - x3 <= -2;
e5:  -x2 - x4 <= -2;
e6:  -x3 - x4 <= -1.5;
e7:  x1 + x2 <= 7.5;
e8:  x1 + x3 <= 7.5;
e9:  x1 + x4 <= 7.5;
e10:  x2 + x3 <= 7;
e11:  x2 + x4 <= 7;
e12:  x3 + x4 <= 6.5;
e13:  x1 + x2 + x3 + x4 >= 5;
