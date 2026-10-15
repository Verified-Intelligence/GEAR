# NLP written by GAMS Convert at 03/06/26 16:51:09
#
# Equation counts
#     Total        E        G        L        N        X        C        B
#         0        0        0        0        0        0        0        0
#
# Variable counts
#                  x        b        i      s1s      s2s       sc       si
#     Total     cont   binary  integer     sos1     sos2    scont     sint
#        30       30        0        0        0        0        0        0
# FX      0
#
# Nonzero counts
#     Total    const       NL
#         0        0        0
#
# Reformulation has removed 1 variable and 1 equation

var x1 := 1.07911905633263, >= 0, <= 6.28318530717959;
var x2 := 5.2984009897393, >= 0, <= 6.28318530717959;
var x3 := 3.45811035025293, >= 0, <= 6.28318530717959;
var x4 := 1.89210525384766, >= 0, <= 6.28318530717959;
var x5 := 1.83602288011424, >= 0, <= 6.28318530717959;
var x6 := 1.40776568196586, >= 0, <= 6.28318530717959;
var x7 := 2.19804988273603, >= 0, <= 6.28318530717959;
var x8 := 5.38010526324397, >= 0, <= 6.28318530717959;
var x9 := 0.421687958263721, >= 0, <= 6.28318530717959;
var x10 := 3.14291632595527, >= 0, <= 6.28318530717959;
var x11 := 6.27135800880335, >= 0, <= 6.28318530717959;
var x12 := 3.63628905742401, >= 0, <= 6.28318530717959;
var x13 := 6.22747254810505, >= 0, <= 6.28318530717959;
var x14 := 4.78936093464518, >= 0, <= 6.28318530717959;
var x15 := 0.821165088944418, >= 0, <= 6.28318530717959;
var x16 := 2.00973575363798, >= 0, <= 3.14159265358979;
var x17 := 0.501140149658736, >= 0, <= 3.14159265358979;
var x18 := 0.78565116527862, >= 0, <= 3.14159265358979;
var x19 := 2.10150120381044, >= 0, <= 3.14159265358979;
var x20 := 1.36771240824304, >= 0, <= 3.14159265358979;
var x21 := 1.13003171315989, >= 0, <= 3.14159265358979;
var x22 := 1.10408561987635, >= 0, <= 3.14159265358979;
var x23 := 0.413093013152841, >= 0, <= 3.14159265358979;
var x24 := 0.471558674471493, >= 0, <= 3.14159265358979;
var x25 := 1.85075511496947, >= 0, <= 3.14159265358979;
var x26 := 2.61032675409977, >= 0, <= 3.14159265358979;
var x27 := 0.725129026833707, >= 0, <= 3.14159265358979;
var x28 := 2.09146648877757, >= 0, <= 3.14159265358979;
var x29 := 2.43742855524136, >= 0, <= 3.14159265358979;
var x30 := 0.953971240543465, >= 0, <= 3.14159265358979;

minimize obj: -log((cos(x1) * sin(x16) - cos(x2) * sin(x17))^2 + (sin(x1) *
    sin(x16) - sin(x2) * sin(x17))^2 + (cos(x16) - cos(x17))^2 + 1e-05) - log((
    cos(x1) * sin(x16) - cos(x3) * sin(x18))^2 + (sin(x1) * sin(x16) - sin(x3)
    * sin(x18))^2 + (cos(x16) - cos(x18))^2 + 1e-05) - log((cos(x1) * sin(x16)
    - cos(x4) * sin(x19))^2 + (sin(x1) * sin(x16) - sin(x4) * sin(x19))^2 + (
    cos(x16) - cos(x19))^2 + 1e-05) - log((cos(x1) * sin(x16) - cos(x5) * sin(
    x20))^2 + (sin(x1) * sin(x16) - sin(x5) * sin(x20))^2 + (cos(x16) - cos(x20))
    ^2 + 1e-05) - log((cos(x1) * sin(x16) - cos(x6) * sin(x21))^2 + (sin(x1) *
    sin(x16) - sin(x6) * sin(x21))^2 + (cos(x16) - cos(x21))^2 + 1e-05) - log((
    cos(x1) * sin(x16) - cos(x7) * sin(x22))^2 + (sin(x1) * sin(x16) - sin(x7)
    * sin(x22))^2 + (cos(x16) - cos(x22))^2 + 1e-05) - log((cos(x1) * sin(x16)
    - cos(x8) * sin(x23))^2 + (sin(x1) * sin(x16) - sin(x8) * sin(x23))^2 + (
    cos(x16) - cos(x23))^2 + 1e-05) - log((cos(x1) * sin(x16) - cos(x9) * sin(
    x24))^2 + (sin(x1) * sin(x16) - sin(x9) * sin(x24))^2 + (cos(x16) - cos(x24))
    ^2 + 1e-05) - log((cos(x1) * sin(x16) - cos(x10) * sin(x25))^2 + (sin(x1)
    * sin(x16) - sin(x10) * sin(x25))^2 + (cos(x16) - cos(x25))^2 + 1e-05) -
    log((cos(x1) * sin(x16) - cos(x11) * sin(x26))^2 + (sin(x1) * sin(x16) -
    sin(x11) * sin(x26))^2 + (cos(x16) - cos(x26))^2 + 1e-05) - log((cos(x1) *
    sin(x16) - cos(x12) * sin(x27))^2 + (sin(x1) * sin(x16) - sin(x12) * sin(
    x27))^2 + (cos(x16) - cos(x27))^2 + 1e-05) - log((cos(x1) * sin(x16) - cos(
    x13) * sin(x28))^2 + (sin(x1) * sin(x16) - sin(x13) * sin(x28))^2 + (cos(
    x16) - cos(x28))^2 + 1e-05) - log((cos(x1) * sin(x16) - cos(x14) * sin(x29))
    ^2 + (sin(x1) * sin(x16) - sin(x14) * sin(x29))^2 + (cos(x16) - cos(x29))^2
    + 1e-05) - log((cos(x1) * sin(x16) - cos(x15) * sin(x30))^2 + (sin(x1) *
    sin(x16) - sin(x15) * sin(x30))^2 + (cos(x16) - cos(x30))^2 + 1e-05) - log(
    (cos(x2) * sin(x17) - cos(x3) * sin(x18))^2 + (sin(x2) * sin(x17) - sin(x3)
    * sin(x18))^2 + (cos(x17) - cos(x18))^2 + 1e-05) - log((cos(x2) * sin(x17)
    - cos(x4) * sin(x19))^2 + (sin(x2) * sin(x17) - sin(x4) * sin(x19))^2 + (
    cos(x17) - cos(x19))^2 + 1e-05) - log((cos(x2) * sin(x17) - cos(x5) * sin(
    x20))^2 + (sin(x2) * sin(x17) - sin(x5) * sin(x20))^2 + (cos(x17) - cos(x20))
    ^2 + 1e-05) - log((cos(x2) * sin(x17) - cos(x6) * sin(x21))^2 + (sin(x2) *
    sin(x17) - sin(x6) * sin(x21))^2 + (cos(x17) - cos(x21))^2 + 1e-05) - log((
    cos(x2) * sin(x17) - cos(x7) * sin(x22))^2 + (sin(x2) * sin(x17) - sin(x7)
    * sin(x22))^2 + (cos(x17) - cos(x22))^2 + 1e-05) - log((cos(x2) * sin(x17)
    - cos(x8) * sin(x23))^2 + (sin(x2) * sin(x17) - sin(x8) * sin(x23))^2 + (
    cos(x17) - cos(x23))^2 + 1e-05) - log((cos(x2) * sin(x17) - cos(x9) * sin(
    x24))^2 + (sin(x2) * sin(x17) - sin(x9) * sin(x24))^2 + (cos(x17) - cos(x24))
    ^2 + 1e-05) - log((cos(x2) * sin(x17) - cos(x10) * sin(x25))^2 + (sin(x2)
    * sin(x17) - sin(x10) * sin(x25))^2 + (cos(x17) - cos(x25))^2 + 1e-05) -
    log((cos(x2) * sin(x17) - cos(x11) * sin(x26))^2 + (sin(x2) * sin(x17) -
    sin(x11) * sin(x26))^2 + (cos(x17) - cos(x26))^2 + 1e-05) - log((cos(x2) *
    sin(x17) - cos(x12) * sin(x27))^2 + (sin(x2) * sin(x17) - sin(x12) * sin(
    x27))^2 + (cos(x17) - cos(x27))^2 + 1e-05) - log((cos(x2) * sin(x17) - cos(
    x13) * sin(x28))^2 + (sin(x2) * sin(x17) - sin(x13) * sin(x28))^2 + (cos(
    x17) - cos(x28))^2 + 1e-05) - log((cos(x2) * sin(x17) - cos(x14) * sin(x29))
    ^2 + (sin(x2) * sin(x17) - sin(x14) * sin(x29))^2 + (cos(x17) - cos(x29))^2
    + 1e-05) - log((cos(x2) * sin(x17) - cos(x15) * sin(x30))^2 + (sin(x2) *
    sin(x17) - sin(x15) * sin(x30))^2 + (cos(x17) - cos(x30))^2 + 1e-05) - log(
    (cos(x3) * sin(x18) - cos(x4) * sin(x19))^2 + (sin(x3) * sin(x18) - sin(x4)
    * sin(x19))^2 + (cos(x18) - cos(x19))^2 + 1e-05) - log((cos(x3) * sin(x18)
    - cos(x5) * sin(x20))^2 + (sin(x3) * sin(x18) - sin(x5) * sin(x20))^2 + (
    cos(x18) - cos(x20))^2 + 1e-05) - log((cos(x3) * sin(x18) - cos(x6) * sin(
    x21))^2 + (sin(x3) * sin(x18) - sin(x6) * sin(x21))^2 + (cos(x18) - cos(x21))
    ^2 + 1e-05) - log((cos(x3) * sin(x18) - cos(x7) * sin(x22))^2 + (sin(x3) *
    sin(x18) - sin(x7) * sin(x22))^2 + (cos(x18) - cos(x22))^2 + 1e-05) - log((
    cos(x3) * sin(x18) - cos(x8) * sin(x23))^2 + (sin(x3) * sin(x18) - sin(x8)
    * sin(x23))^2 + (cos(x18) - cos(x23))^2 + 1e-05) - log((cos(x3) * sin(x18)
    - cos(x9) * sin(x24))^2 + (sin(x3) * sin(x18) - sin(x9) * sin(x24))^2 + (
    cos(x18) - cos(x24))^2 + 1e-05) - log((cos(x3) * sin(x18) - cos(x10) * sin(
    x25))^2 + (sin(x3) * sin(x18) - sin(x10) * sin(x25))^2 + (cos(x18) - cos(
    x25))^2 + 1e-05) - log((cos(x3) * sin(x18) - cos(x11) * sin(x26))^2 + (sin(
    x3) * sin(x18) - sin(x11) * sin(x26))^2 + (cos(x18) - cos(x26))^2 + 1e-05)
    - log((cos(x3) * sin(x18) - cos(x12) * sin(x27))^2 + (sin(x3) * sin(x18)
    - sin(x12) * sin(x27))^2 + (cos(x18) - cos(x27))^2 + 1e-05) - log((cos(x3)
    * sin(x18) - cos(x13) * sin(x28))^2 + (sin(x3) * sin(x18) - sin(x13) *
    sin(x28))^2 + (cos(x18) - cos(x28))^2 + 1e-05) - log((cos(x3) * sin(x18) -
    cos(x14) * sin(x29))^2 + (sin(x3) * sin(x18) - sin(x14) * sin(x29))^2 + (
    cos(x18) - cos(x29))^2 + 1e-05) - log((cos(x3) * sin(x18) - cos(x15) * sin(
    x30))^2 + (sin(x3) * sin(x18) - sin(x15) * sin(x30))^2 + (cos(x18) - cos(
    x30))^2 + 1e-05) - log((cos(x4) * sin(x19) - cos(x5) * sin(x20))^2 + (sin(
    x4) * sin(x19) - sin(x5) * sin(x20))^2 + (cos(x19) - cos(x20))^2 + 1e-05)
    - log((cos(x4) * sin(x19) - cos(x6) * sin(x21))^2 + (sin(x4) * sin(x19) -
    sin(x6) * sin(x21))^2 + (cos(x19) - cos(x21))^2 + 1e-05) - log((cos(x4) *
    sin(x19) - cos(x7) * sin(x22))^2 + (sin(x4) * sin(x19) - sin(x7) * sin(x22))
    ^2 + (cos(x19) - cos(x22))^2 + 1e-05) - log((cos(x4) * sin(x19) - cos(x8)
    * sin(x23))^2 + (sin(x4) * sin(x19) - sin(x8) * sin(x23))^2 + (cos(x19) -
    cos(x23))^2 + 1e-05) - log((cos(x4) * sin(x19) - cos(x9) * sin(x24))^2 + (
    sin(x4) * sin(x19) - sin(x9) * sin(x24))^2 + (cos(x19) - cos(x24))^2 +
    1e-05) - log((cos(x4) * sin(x19) - cos(x10) * sin(x25))^2 + (sin(x4) * sin(
    x19) - sin(x10) * sin(x25))^2 + (cos(x19) - cos(x25))^2 + 1e-05) - log((
    cos(x4) * sin(x19) - cos(x11) * sin(x26))^2 + (sin(x4) * sin(x19) - sin(x11)
    * sin(x26))^2 + (cos(x19) - cos(x26))^2 + 1e-05) - log((cos(x4) * sin(x19)
    - cos(x12) * sin(x27))^2 + (sin(x4) * sin(x19) - sin(x12) * sin(x27))^2 +
    (cos(x19) - cos(x27))^2 + 1e-05) - log((cos(x4) * sin(x19) - cos(x13) *
    sin(x28))^2 + (sin(x4) * sin(x19) - sin(x13) * sin(x28))^2 + (cos(x19) -
    cos(x28))^2 + 1e-05) - log((cos(x4) * sin(x19) - cos(x14) * sin(x29))^2 + (
    sin(x4) * sin(x19) - sin(x14) * sin(x29))^2 + (cos(x19) - cos(x29))^2 +
    1e-05) - log((cos(x4) * sin(x19) - cos(x15) * sin(x30))^2 + (sin(x4) * sin(
    x19) - sin(x15) * sin(x30))^2 + (cos(x19) - cos(x30))^2 + 1e-05) - log((
    cos(x5) * sin(x20) - cos(x6) * sin(x21))^2 + (sin(x5) * sin(x20) - sin(x6)
    * sin(x21))^2 + (cos(x20) - cos(x21))^2 + 1e-05) - log((cos(x5) * sin(x20)
    - cos(x7) * sin(x22))^2 + (sin(x5) * sin(x20) - sin(x7) * sin(x22))^2 + (
    cos(x20) - cos(x22))^2 + 1e-05) - log((cos(x5) * sin(x20) - cos(x8) * sin(
    x23))^2 + (sin(x5) * sin(x20) - sin(x8) * sin(x23))^2 + (cos(x20) - cos(x23))
    ^2 + 1e-05) - log((cos(x5) * sin(x20) - cos(x9) * sin(x24))^2 + (sin(x5) *
    sin(x20) - sin(x9) * sin(x24))^2 + (cos(x20) - cos(x24))^2 + 1e-05) - log((
    cos(x5) * sin(x20) - cos(x10) * sin(x25))^2 + (sin(x5) * sin(x20) - sin(x10)
    * sin(x25))^2 + (cos(x20) - cos(x25))^2 + 1e-05) - log((cos(x5) * sin(x20)
    - cos(x11) * sin(x26))^2 + (sin(x5) * sin(x20) - sin(x11) * sin(x26))^2 +
    (cos(x20) - cos(x26))^2 + 1e-05) - log((cos(x5) * sin(x20) - cos(x12) *
    sin(x27))^2 + (sin(x5) * sin(x20) - sin(x12) * sin(x27))^2 + (cos(x20) -
    cos(x27))^2 + 1e-05) - log((cos(x5) * sin(x20) - cos(x13) * sin(x28))^2 + (
    sin(x5) * sin(x20) - sin(x13) * sin(x28))^2 + (cos(x20) - cos(x28))^2 +
    1e-05) - log((cos(x5) * sin(x20) - cos(x14) * sin(x29))^2 + (sin(x5) * sin(
    x20) - sin(x14) * sin(x29))^2 + (cos(x20) - cos(x29))^2 + 1e-05) - log((
    cos(x5) * sin(x20) - cos(x15) * sin(x30))^2 + (sin(x5) * sin(x20) - sin(x15)
    * sin(x30))^2 + (cos(x20) - cos(x30))^2 + 1e-05) - log((cos(x6) * sin(x21)
    - cos(x7) * sin(x22))^2 + (sin(x6) * sin(x21) - sin(x7) * sin(x22))^2 + (
    cos(x21) - cos(x22))^2 + 1e-05) - log((cos(x6) * sin(x21) - cos(x8) * sin(
    x23))^2 + (sin(x6) * sin(x21) - sin(x8) * sin(x23))^2 + (cos(x21) - cos(x23))
    ^2 + 1e-05) - log((cos(x6) * sin(x21) - cos(x9) * sin(x24))^2 + (sin(x6) *
    sin(x21) - sin(x9) * sin(x24))^2 + (cos(x21) - cos(x24))^2 + 1e-05) - log((
    cos(x6) * sin(x21) - cos(x10) * sin(x25))^2 + (sin(x6) * sin(x21) - sin(x10)
    * sin(x25))^2 + (cos(x21) - cos(x25))^2 + 1e-05) - log((cos(x6) * sin(x21)
    - cos(x11) * sin(x26))^2 + (sin(x6) * sin(x21) - sin(x11) * sin(x26))^2 +
    (cos(x21) - cos(x26))^2 + 1e-05) - log((cos(x6) * sin(x21) - cos(x12) *
    sin(x27))^2 + (sin(x6) * sin(x21) - sin(x12) * sin(x27))^2 + (cos(x21) -
    cos(x27))^2 + 1e-05) - log((cos(x6) * sin(x21) - cos(x13) * sin(x28))^2 + (
    sin(x6) * sin(x21) - sin(x13) * sin(x28))^2 + (cos(x21) - cos(x28))^2 +
    1e-05) - log((cos(x6) * sin(x21) - cos(x14) * sin(x29))^2 + (sin(x6) * sin(
    x21) - sin(x14) * sin(x29))^2 + (cos(x21) - cos(x29))^2 + 1e-05) - log((
    cos(x6) * sin(x21) - cos(x15) * sin(x30))^2 + (sin(x6) * sin(x21) - sin(x15)
    * sin(x30))^2 + (cos(x21) - cos(x30))^2 + 1e-05) - log((cos(x7) * sin(x22)
    - cos(x8) * sin(x23))^2 + (sin(x7) * sin(x22) - sin(x8) * sin(x23))^2 + (
    cos(x22) - cos(x23))^2 + 1e-05) - log((cos(x7) * sin(x22) - cos(x9) * sin(
    x24))^2 + (sin(x7) * sin(x22) - sin(x9) * sin(x24))^2 + (cos(x22) - cos(x24))
    ^2 + 1e-05) - log((cos(x7) * sin(x22) - cos(x10) * sin(x25))^2 + (sin(x7)
    * sin(x22) - sin(x10) * sin(x25))^2 + (cos(x22) - cos(x25))^2 + 1e-05) -
    log((cos(x7) * sin(x22) - cos(x11) * sin(x26))^2 + (sin(x7) * sin(x22) -
    sin(x11) * sin(x26))^2 + (cos(x22) - cos(x26))^2 + 1e-05) - log((cos(x7) *
    sin(x22) - cos(x12) * sin(x27))^2 + (sin(x7) * sin(x22) - sin(x12) * sin(
    x27))^2 + (cos(x22) - cos(x27))^2 + 1e-05) - log((cos(x7) * sin(x22) - cos(
    x13) * sin(x28))^2 + (sin(x7) * sin(x22) - sin(x13) * sin(x28))^2 + (cos(
    x22) - cos(x28))^2 + 1e-05) - log((cos(x7) * sin(x22) - cos(x14) * sin(x29))
    ^2 + (sin(x7) * sin(x22) - sin(x14) * sin(x29))^2 + (cos(x22) - cos(x29))^2
    + 1e-05) - log((cos(x7) * sin(x22) - cos(x15) * sin(x30))^2 + (sin(x7) *
    sin(x22) - sin(x15) * sin(x30))^2 + (cos(x22) - cos(x30))^2 + 1e-05) - log(
    (cos(x8) * sin(x23) - cos(x9) * sin(x24))^2 + (sin(x8) * sin(x23) - sin(x9)
    * sin(x24))^2 + (cos(x23) - cos(x24))^2 + 1e-05) - log((cos(x8) * sin(x23)
    - cos(x10) * sin(x25))^2 + (sin(x8) * sin(x23) - sin(x10) * sin(x25))^2 +
    (cos(x23) - cos(x25))^2 + 1e-05) - log((cos(x8) * sin(x23) - cos(x11) *
    sin(x26))^2 + (sin(x8) * sin(x23) - sin(x11) * sin(x26))^2 + (cos(x23) -
    cos(x26))^2 + 1e-05) - log((cos(x8) * sin(x23) - cos(x12) * sin(x27))^2 + (
    sin(x8) * sin(x23) - sin(x12) * sin(x27))^2 + (cos(x23) - cos(x27))^2 +
    1e-05) - log((cos(x8) * sin(x23) - cos(x13) * sin(x28))^2 + (sin(x8) * sin(
    x23) - sin(x13) * sin(x28))^2 + (cos(x23) - cos(x28))^2 + 1e-05) - log((
    cos(x8) * sin(x23) - cos(x14) * sin(x29))^2 + (sin(x8) * sin(x23) - sin(x14)
    * sin(x29))^2 + (cos(x23) - cos(x29))^2 + 1e-05) - log((cos(x8) * sin(x23)
    - cos(x15) * sin(x30))^2 + (sin(x8) * sin(x23) - sin(x15) * sin(x30))^2 +
    (cos(x23) - cos(x30))^2 + 1e-05) - log((cos(x9) * sin(x24) - cos(x10) *
    sin(x25))^2 + (sin(x9) * sin(x24) - sin(x10) * sin(x25))^2 + (cos(x24) -
    cos(x25))^2 + 1e-05) - log((cos(x9) * sin(x24) - cos(x11) * sin(x26))^2 + (
    sin(x9) * sin(x24) - sin(x11) * sin(x26))^2 + (cos(x24) - cos(x26))^2 +
    1e-05) - log((cos(x9) * sin(x24) - cos(x12) * sin(x27))^2 + (sin(x9) * sin(
    x24) - sin(x12) * sin(x27))^2 + (cos(x24) - cos(x27))^2 + 1e-05) - log((
    cos(x9) * sin(x24) - cos(x13) * sin(x28))^2 + (sin(x9) * sin(x24) - sin(x13)
    * sin(x28))^2 + (cos(x24) - cos(x28))^2 + 1e-05) - log((cos(x9) * sin(x24)
    - cos(x14) * sin(x29))^2 + (sin(x9) * sin(x24) - sin(x14) * sin(x29))^2 +
    (cos(x24) - cos(x29))^2 + 1e-05) - log((cos(x9) * sin(x24) - cos(x15) *
    sin(x30))^2 + (sin(x9) * sin(x24) - sin(x15) * sin(x30))^2 + (cos(x24) -
    cos(x30))^2 + 1e-05) - log((cos(x10) * sin(x25) - cos(x11) * sin(x26))^2 +
    (sin(x10) * sin(x25) - sin(x11) * sin(x26))^2 + (cos(x25) - cos(x26))^2 +
    1e-05) - log((cos(x10) * sin(x25) - cos(x12) * sin(x27))^2 + (sin(x10) *
    sin(x25) - sin(x12) * sin(x27))^2 + (cos(x25) - cos(x27))^2 + 1e-05) - log(
    (cos(x10) * sin(x25) - cos(x13) * sin(x28))^2 + (sin(x10) * sin(x25) - sin(
    x13) * sin(x28))^2 + (cos(x25) - cos(x28))^2 + 1e-05) - log((cos(x10) *
    sin(x25) - cos(x14) * sin(x29))^2 + (sin(x10) * sin(x25) - sin(x14) * sin(
    x29))^2 + (cos(x25) - cos(x29))^2 + 1e-05) - log((cos(x10) * sin(x25) -
    cos(x15) * sin(x30))^2 + (sin(x10) * sin(x25) - sin(x15) * sin(x30))^2 + (
    cos(x25) - cos(x30))^2 + 1e-05) - log((cos(x11) * sin(x26) - cos(x12) *
    sin(x27))^2 + (sin(x11) * sin(x26) - sin(x12) * sin(x27))^2 + (cos(x26) -
    cos(x27))^2 + 1e-05) - log((cos(x11) * sin(x26) - cos(x13) * sin(x28))^2 +
    (sin(x11) * sin(x26) - sin(x13) * sin(x28))^2 + (cos(x26) - cos(x28))^2 +
    1e-05) - log((cos(x11) * sin(x26) - cos(x14) * sin(x29))^2 + (sin(x11) *
    sin(x26) - sin(x14) * sin(x29))^2 + (cos(x26) - cos(x29))^2 + 1e-05) - log(
    (cos(x11) * sin(x26) - cos(x15) * sin(x30))^2 + (sin(x11) * sin(x26) - sin(
    x15) * sin(x30))^2 + (cos(x26) - cos(x30))^2 + 1e-05) - log((cos(x12) *
    sin(x27) - cos(x13) * sin(x28))^2 + (sin(x12) * sin(x27) - sin(x13) * sin(
    x28))^2 + (cos(x27) - cos(x28))^2 + 1e-05) - log((cos(x12) * sin(x27) -
    cos(x14) * sin(x29))^2 + (sin(x12) * sin(x27) - sin(x14) * sin(x29))^2 + (
    cos(x27) - cos(x29))^2 + 1e-05) - log((cos(x12) * sin(x27) - cos(x15) *
    sin(x30))^2 + (sin(x12) * sin(x27) - sin(x15) * sin(x30))^2 + (cos(x27) -
    cos(x30))^2 + 1e-05) - log((cos(x13) * sin(x28) - cos(x14) * sin(x29))^2 +
    (sin(x13) * sin(x28) - sin(x14) * sin(x29))^2 + (cos(x28) - cos(x29))^2 +
    1e-05) - log((cos(x13) * sin(x28) - cos(x15) * sin(x30))^2 + (sin(x13) *
    sin(x28) - sin(x15) * sin(x30))^2 + (cos(x28) - cos(x30))^2 + 1e-05) - log(
    (cos(x14) * sin(x29) - cos(x15) * sin(x30))^2 + (sin(x14) * sin(x29) - sin(
    x15) * sin(x30))^2 + (cos(x29) - cos(x30))^2 + 1e-05);

