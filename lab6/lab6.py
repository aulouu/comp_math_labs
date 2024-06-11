import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable

def f1(x, y):
    return x ** 2 + x - 2 * y

def f1_ac(x):
    return x ** 2 / 2

def f2(x, y):
    return 2 * x - y + x ** 2

def f2_ac(x):
    return x ** 2

def f3(x, y):
    return 5 * x ** 2 - 2 * y / x

def f3_ac(x):
    return x ** 3

def euler(f, x0, y0, xn, h):
    x = np.arange(x0, xn + h, h)
    y = np.zeros(len(x))
    y[0] = y0
    for i in range(len(x) - 1):
        y[i + 1] = y[i] + h /2 * (f(x[i], y[i]) + f(x[i + 1], y[i] + h * f(x[i], y[i])))
    return x, y, h

def rk4(f, x0, y0, xn, h):
    x = np.arange(x0, xn + h, h)
    y = np.zeros(len(x))
    y[0] = y0
    for i in range(len(x) - 1):
        k1 = h * f(x[i], y[i])
        k2 = h * f(x[i] + h / 2, y[i] + k1 / 2)
        k3 = h * f(x[i] + h / 2, y[i] + k2 / 2)
        k4 = h * f(x[i] + h, y[i] + k3)
        y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return x, y, h

def milne(f, x0, y0, xn, h, eps):
    x = np.arange(x0, xn + h, h)
    y = np.zeros(len(x))
    y[0] = y0
    y[1] = y_approx_rk4[np.argmin(np.abs(x_approx_rk4 - (x_approx_rk4[0] + h)))]
    y[2] = y_approx_rk4[np.argmin(np.abs(x_approx_rk4 - (x_approx_rk4[0] + 2 * h)))]
    y[3] = y_approx_rk4[np.argmin(np.abs(x_approx_rk4 - (x_approx_rk4[0] + 3 * h)))]
    for i in range(3, len(x) - 1):
        y_pred = y[i - 3] + 4 * h * (2 * f(x[i - 2], y[i - 2]) - f(x[i - 1], y[i - 1]) + 2 * f(x[i], y[i])) / 3
        y_corr = y[i - 1] + h * (f(x[i - 1], y[i - 1]) + 4 * f(x[i], y[i]) + f(x[i + 1], y_pred)) / 3
        while np.abs(y_corr - y_pred) > eps:
            y_pred = y_corr
            y_corr = y[i - 1] + h * (f(x[i - 1], y[i - 1]) + 4 * f(x[i], y[i]) + f(x[i + 1], y_pred)) / 3
        y[i + 1] = y_corr
    return x, y

print("Выберите уравнение:")
print("1. y' = x^2 + x - 2y")
print("2. y' = 2x - y + x^2")
print("3. y' = 5x^2 - 2y/x")
eq = int(input())
if eq == 1:
    f = f1
    f_ac = f1_ac
elif eq == 2:
    f = f2
    f_ac = f2_ac
elif eq == 3:
    f = f3
    f_ac = f3_ac
else:
    print("Некорректный выбор уравнения.")
    exit()

x0 = float(input("Введите начальное значение x0: "))
y0 = float(input("Введите начальное условие y0: "))
xn = float(input("Введите конечное значение xn: "))
h = float(input("Введите шаг h: "))
eps = float(input("Введите точность eps: "))

# Метод Эйлера
h_euler = h
x_approx_euler, y_approx_euler, h1_euler = euler(f, x0, y0, xn, h_euler)
h_init_euler = h1_euler
x_approx_euler_1, y_approx_euler_1, h2_euler = euler(f, x0, y0, xn, h_euler/2)
# print("Эйлер:\n", h1_euler, y_approx_euler[-1])
# print(h2_euler, y_approx_euler_1[-1])
while np.abs(y_approx_euler[-1] - y_approx_euler_1[-1]) / 3 > eps:
    x_approx_euler, y_approx_euler, h1_euler = x_approx_euler_1, y_approx_euler_1, h2_euler
    h_euler /= 2
    x_approx_euler_1, y_approx_euler_1, h2_euler = euler(f, x0, y0, xn, h_euler/2)
# print(h1_euler, y_approx_euler[-1])
x_approx_euler, y_approx_euler, h2_euler = x_approx_euler_1, y_approx_euler_1, h2_euler
# print(h2_euler, y_approx_euler_1[-1])

# Метод Рунге-Кутта 4-го порядка
h_rk4 = h
x_approx_rk4, y_approx_rk4, h1_rk4 = rk4(f, x0, y0, xn, h_rk4)
h_init_rk4 = h1_rk4
x_approx_rk4_1, y_approx_rk4_1, h2_rk4 = rk4(f, x0, y0, xn, h_rk4 / 2)
while np.abs(y_approx_rk4[-1] - y_approx_rk4_1[-1]) / 15 > eps:
    x_approx_rk4, y_approx_rk4, h1_rk4 = x_approx_rk4_1, y_approx_rk4_1, h2_rk4
    h_rk4 /= 2
    x_approx_rk4_1, y_approx_rk4_1, h2_rk4 = rk4(f, x0, y0, xn, h_rk4 / 2)
# print("Рунге-Кутта:\n", h1_rk4, y_approx_rk4[-1])
x_approx_rk4, y_approx_rk4, h2_rk4 = x_approx_rk4_1, y_approx_rk4_1, h2_rk4
# print(h2_rk4, y_approx_rk4_1[-1])

# Метод Милна
h_milne = h
x_approx_milne, y_approx_milne = milne(f, x0, y0, xn, h_milne, eps)

x_true = np.linspace(x0, xn, 1000)
y_true = f_ac(x_true)

# Таблица для метода Эйлера
table_euler = PrettyTable(["x", "y_approx", "y_true", "Error"])
x_current_euler = x_approx_euler[0]
for x, y in zip(x_approx_euler, y_approx_euler):
    if np.round(x, 6) == np.round(x_current_euler, 6):
        y_true_value = f_ac(x)
        error = np.abs(y - y_true_value)
        table_euler.add_row([round(x, 6), round(y, 6), round(y_true_value, 6), round(error, 6)])
        x_current_euler += h_init_euler
print("\nТаблица приближенных значений (метод Эйлера):")
print(table_euler)
print(f"Решение с шагом {h2_euler}")

# Таблица для метода Рунге-Кутта 4-го порядка
table_rk4 = PrettyTable(["x", "y_approx", "y_true", "Error"])
x_current_rk4 = x_approx_rk4[0]
for x, y in zip(x_approx_rk4, y_approx_rk4):
    if np.round(x, 6) == np.round(x_current_rk4, 6):
        y_true_value = f_ac(x)
        error = np.abs(y - y_true_value)
        table_rk4.add_row([round(x, 6), round(y, 6), round(y_true_value, 6), round(error, 6)])
        x_current_rk4 += h_init_rk4
print("\nТаблица приближенных значений (метод Рунге-Кутта 4-го порядка):")
print(table_rk4)
print(f"Решение с шагом {h2_rk4}")

# Таблица для метода Милна
table_milne = PrettyTable(["x", "y_approx", "y_true", "Error"])
for x, y in zip(x_approx_milne, y_approx_milne):
    y_true_value = f_ac(x)
    error = np.abs(y - y_true_value)
    table_milne.add_row([round(x, 6), round(y, 6), round(y_true_value, 6), round(error, 6)])
print("\nТаблица приближенных значений (метод Милна):")
print(table_milne)
print(f"Решение с шагом {h_milne}")

plt.plot(x_approx_euler, y_approx_euler, label="Метод Эйлера")
plt.plot(x_approx_rk4, y_approx_rk4, label="Метод Рунге-Кутта 4-го порядка")
plt.plot(x_approx_milne, y_approx_milne, label="Метод Милна")
plt.plot(x_true, y_true, label="Точное решение", linestyle='--')
plt.legend()
plt.show()
