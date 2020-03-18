from _pydecimal import Decimal
from scipy.stats import f, t
from random import randrange
from math import sqrt, fabs


def det(a):
    from numpy.linalg import det
    return det(a)


class Critical_values:
    @staticmethod
    def get_cohren_value(size_of_selections, qty_of_selections, significance):
        size_of_selections += 1
        partResult1 = significance / (size_of_selections - 1)
        params = [partResult1, qty_of_selections, (size_of_selections - 1 - 1) * qty_of_selections]
        fisher = f.isf(*params)
        result = fisher / (fisher + (size_of_selections - 1 - 1))
        return Decimal(result).quantize(Decimal('.0001')).__float__()

    @staticmethod
    def get_student_value(f3, significance):
        return Decimal(abs(t.ppf(significance / 2, f3))).quantize(Decimal('.0001')).__float__()

    @staticmethod
    def get_fisher_value(f3, f4, significance):
        return Decimal(abs(f.isf(significance, f4, f3))).quantize(Decimal('.0001')).__float__()


print("--Рівняння регресії матиме вигляд: ŷ = b0 + b1*X1 + b2*X2 + b3*X3--")
print("--Матриця планування експеременту--")
matrix_pfe = [[1, -1, -1, -1], [1, -1, 1, 1], [1, 1, -1, 1], [1, 1, 1, -1]]
for i in range(len(matrix_pfe)):
    print("|", end=" ")
    for j in range(len(matrix_pfe[i])):
        print(matrix_pfe[i][j], end=" ")
    print("|")
x1_min = 10
x1_max = 40
x2_min = 15
x2_max = 50
x3_min = 10
x3_max = 30
y_min = int((x1_min + x2_min + x3_min) / 3)
y_max = int((x1_max + x2_max + x3_max) / 3)
matrix = [[x1_min, x2_min, x3_min], [x1_min, x2_max, x3_max],
          [x1_max, x2_min, x3_max], [x1_max, x2_max, x3_min]]
for i in range(len(matrix)):
    for j in range(3):
        matrix[i].append(randrange(y_min, y_max))

print("--Матриця з натуральних значень факторів--")
for i in range(len(matrix)):
    print("|", end=" ")
    for j in range(len(matrix[i])):
        print(matrix[i][j], end=" ")
    print("|")

my1 = sum(matrix[0][3:]) / 3
my2 = sum(matrix[1][3:]) / 3
my3 = sum(matrix[2][3:]) / 3
my4 = sum(matrix[3][3:]) / 3
my = (my1 + my2 + my3 + my4) / 4
mx1 = (matrix[0][0] + matrix[1][0] + matrix[2][0] + matrix[3][0]) / 4
mx2 = (matrix[0][1] + matrix[1][1] + matrix[2][1] + matrix[3][1]) / 4
mx3 = (matrix[0][2] + matrix[1][2] + matrix[2][2] + matrix[3][2]) / 4
a1 = (matrix[0][0] * my1 + matrix[1][0] * my2 + matrix[2][0] * my3 + matrix[3][0] * my4) / 4
a2 = (matrix[0][1] * my1 + matrix[1][1] * my2 + matrix[2][1] * my3 + matrix[3][1] * my4) / 4
a3 = (matrix[0][2] * my1 + matrix[1][2] * my2 + matrix[2][2] * my3 + matrix[3][2] * my4) / 4
a11 = (matrix[0][0]**2 + matrix[1][0]**2 + matrix[2][0]**2 + matrix[3][0]**2) / 4
a22 = (matrix[0][1]**2 + matrix[1][1]**2 + matrix[2][1]**2 + matrix[3][1]**2) / 4
a33 = (matrix[0][2]**2 + matrix[1][2]**2 + matrix[2][2]**2 + matrix[3][2]**2) / 4
a12 = (matrix[0][0] * matrix[0][1] + matrix[1][0] * matrix[1][1] + matrix[2][0] * matrix[2][1] +
       matrix[3][0] * matrix[3][1]) / 4
a21 = a12
a13 = (matrix[0][0] * matrix[0][2] + matrix[1][0] * matrix[1][2] + matrix[2][0] * matrix[2][2] +
       matrix[3][0] * matrix[3][2]) / 4
a31 = a13
a23 = (matrix[0][1] * matrix[0][2] + matrix[1][1] * matrix[1][2] + matrix[2][1] * matrix[2][2] +
       matrix[3][1] * matrix[3][2]) / 4
a32 = a23

b0_numerator = [[my, mx1, mx2, mx3], [a1, a11, a12, a13], [a2, a21, a22, a23], [a3, a31, a32, a33]]
b1_numerator = [[1, my, mx2, mx3], [mx1, a1, a12, a13], [mx2, a2, a22, a23], [mx3, a3, a32, a33]]
b2_numerator = [[1, mx1, my, mx3], [mx1, a11, a1, a13], [mx2, a21, a2, a23], [mx3, a31, a3, a33]]
b3_numerator = [[1, mx1, mx2, my], [mx1, a11, a12, a1], [mx2, a21, a22, a2], [mx3, a31, a32, a3]]
b_denominator = [[1, mx1, mx2, mx3], [mx1, a11, a12, a13], [mx2, a21, a22, a23], [mx3, a31, a32, a33]]

b0 = det(b0_numerator) / det(b_denominator)
b1 = det(b1_numerator) / det(b_denominator)
b2 = det(b2_numerator) / det(b_denominator)
b3 = det(b3_numerator) / det(b_denominator)

print("--Рівняння регресії--")
print("{:.3f} + {:.3f} * X1 + {:.3f} * X2 + {:.3f} * X3 = ŷ".format(b0, b1, b2, b3))
print("--Перевірка--")

print("{:.3f} + {:.3f} * {:.3f} + {:.3f} * {:.3f} + {:.3f} * {:.3f} = ".format(b0, b1, x1_min, b2, x2_min, b3, x3_min)
      + str(b0 + b1 * x1_min + b2 * x2_min + b3 * x3_min))
print("{:.3f} + {:.3f} * {:.3f} + {:.3f} * {:.3f} + {:.3f} * {:.3f} = ".format(b0, b1, x1_min, b2, x2_max, b3, x3_max)
      + str(b0 + b1 * x1_min + b2 * x2_max + b3 * x3_max))
print("{:.3f} + {:.3f} * {:.3f} + {:.3f} * {:.3f} + {:.3f} * {:.3f} = ".format(b0, b1, x1_max, b2, x2_min, b3, x3_max)
      + str(b0 + b1 * x1_max + b2 * x2_min + b3 * x3_max))
print("{:.3f} + {:.3f} * {:.3f} + {:.3f} * {:.3f} + {:.3f} * {:.3f} = ".format(b0, b1, x1_max, b2, x2_max, b3, x3_min)
      + str(b0 + b1 * x1_max + b2 * x2_max + b3 * x3_min))
dispersion_y1 = ((matrix[0][3] - my1)**2 + (matrix[0][4] - my1)**2 + (matrix[0][5] - my1)** 2) / 3
dispersion_y2 = ((matrix[1][3] - my2)**2 + (matrix[1][4] - my2)**2 + (matrix[1][5] - my2)** 2) / 3
dispersion_y3 = ((matrix[2][3] - my3)**2 + (matrix[2][4] - my3)**2 + (matrix[2][5] - my3)** 2) / 3
dispersion_y4 = ((matrix[3][3] - my4)**2 + (matrix[3][4] - my4)**2 + (matrix[3][5] - my4)** 2) / 3
dispersion_lst = [dispersion_y1, dispersion_y2, dispersion_y3, dispersion_y4]
Gp = max(dispersion_lst) / sum(dispersion_lst)
f1 = 2
f2 = 4
q = 0.05
print("--Критерій Кохрена--")
Gt = Critical_values.get_cohren_value(f2, f1, q)
if Gt > Gp:
    print("--Дисперсія однорідна--")
else:
    print("--Дисперсія не однорідна--")

print("--Критерій Стьюдента--")
S_2b = (dispersion_y1 + dispersion_y2 + dispersion_y3 + dispersion_y4) / 4
S_2b /= 12
S_b = sqrt(S_2b)
beta_0 = (my1 + my2 + my3 + my4) / 4
beta_1 = (-my1 - my2 + my3 + my4) / 4
beta_2 = (-my1 + my2 - my3 + my4) / 4
beta_3 = (-my1 + my2 + my3 - my4) / 4
t_0 = beta_0 / S_b
t_1 = beta_1 / S_b
t_2 = beta_2 / S_b
t_3 = beta_3 / S_b
tt = Critical_values.get_student_value(f1 * f2, q)
t_lst = [fabs(t_0), fabs(t_1), fabs(t_2), fabs(t_3)]
b_lst = [b0, b1, b2, b3]
for i in range(4):
    if t_lst[i] > tt:
        continue
    else:
        t_lst[i] = 0
for j in range(4):
    if t_lst[j] != 0:
        continue
    else:
        b_lst[j] = 0
print("--Перевірка значемих коефіціентів--")
yj1 = b_lst[0] + b_lst[1] * x1_min + b_lst[2] * x2_min + b_lst[3] * x3_min
yj2 = b_lst[0] + b_lst[1] * x1_min + b_lst[2] * x2_max + b_lst[3] * x3_max
yj3 = b_lst[0] + b_lst[1] * x1_max + b_lst[2] * x2_min + b_lst[3] * x3_max
yj4 = b_lst[0] + b_lst[1] * x1_max + b_lst[2] * x2_max + b_lst[3] * x3_min
print("{:.3f} + {:.3f} * {:.3f} + {:.3f} * {:.3f} + {:.3f} * {:.3f} = "
      "".format(b_lst[0], b_lst[1], x1_min, b_lst[2], x2_min, b_lst[3],
                x3_min) + str(yj1))
print("{:.3f} + {:.3f} * {:.3f} + {:.3f} * {:.3f} + {:.3f} * {:.3f} = "
      "".format(b_lst[0], b_lst[1], x1_min, b_lst[2], x2_max, b_lst[3],
                x3_max) + str(yj2))
print("{:.3f} + {:.3f} * {:.3f} + {:.3f} * {:.3f} + {:.3f} * {:.3f} = "
      "".format(b_lst[0], b_lst[1], x1_max, b_lst[2], x2_min, b_lst[3],
                x3_max) + str(yj3))
print("{:.3f} + {:.3f} * {:.3f} + {:.3f} * {:.3f} + {:.3f} * {:.3f} = "
      "".format(b_lst[0], b_lst[1], x1_max, b_lst[2], x2_max, b_lst[3],
                x3_min) + str(yj4))
print("--Критерій Фішера--")
for i in range(3):
    if b_lst[i] == 0:
        del b_lst[i]

d = len(b_lst)
f4 = 4 - d
S_2ad = 3 * ((yj1 - my1)**2 + (yj2 - my2)**2 + (yj3 - my3)**2 + (yj4 - my4)**2) / f4
Fp = S_2ad / S_2b
Ft = Critical_values.get_fisher_value(f1 * f2, f4, q)
if Fp > Ft:
    print("Pівняння регресії неадекватно оригіналу при рівні значимості 0.05")
else:
    print("Pівняння регресії адекватно оригіналу при рівні значимості 0.05")