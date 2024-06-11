import math
import numpy as np
from sympy import diff
import matplotlib.pyplot as plt


def root_exists(f, a0, b0):
    if f(a0) * f(b0) > 0:
        print("На введенном интервале отсутствуют корни уравнения или существует несколько корней")
        return False
    if not mon(f, a0, b0):
        print("На заданном интервале более одного корня")
        return False
    return True


def mon(func, a, b):
    x_values = np.linspace(a, b, num=1000)
    y_values = func(x_values)
    cnt = 0
    for i in range(1, len(x_values)):
        if y_values[i] * y_values[i - 1] < 0:
            cnt += 1
    if cnt > 1 or cnt == 0:
        return False
    elif cnt == 1:
        return True


def simple_graph(f, l, r):
    x = np.arange(l, r, 0.1)
    y = f(x)
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()


def sys_graph(f1, f2, l, r):
    y_values_1 = np.linspace(l, r, 400)
    x_values_1 = np.linspace(l, r, 400)

    z1 = np.zeros((len(x_values_1), len(y_values_1)))

    for i in range(len(x_values_1)):
        for j in range(len(y_values_1)):
            z1[i, j] = f1(y_values_1[j], x_values_1[i])

    y_values_2 = np.linspace(l, r, 400)
    x_values_2 = np.linspace(l, r, 400)

    z2 = np.zeros((len(x_values_2), len(y_values_2)))

    for i in range(len(x_values_2)):
        for j in range(len(y_values_2)):
            z2[i, j] = f2(y_values_2[j], x_values_2[i])

    plt.contour(y_values_1, x_values_1, z1, levels=[0], colors='r')
    plt.contour(y_values_2, x_values_2, z2, levels=[0], colors='b')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.axis('equal')
    plt.show()


def half_division(func, a0, b0, eps):
    a = a0
    b = b0
    cnt = 0
    while True:
        x = (a + b) / 2
        cnt += 1
        if func(x) == 0:
            return x
        elif func(a) * func(x) > 0:
            a = x
        else:
            b = x
        if abs(func(x)) <= eps:
            return x, func(x), cnt


def newton(func, a0, b0, eps):
    if func(a0) * diff2(func, a0) > 0:
        x = a0
    elif func(b0) * diff2(func, b0) > 0:
        x = b0
    else: x = a0
    cnt = 0
    while True:
        x = x - func(x) / diff(func, x)
        cnt += 1
        if abs(func(x)) <= eps:
            return x, func(x), cnt


def simple_iteration(func, a0, b0, eps):
    x = b0
    cnt = 0
    if diff(func, (a0 + b0) / 2) > 0:
        la = -1
    else:
        la = 1
    la *= 1 / max(abs(diff(func, a0)), abs(diff(func, b0)))

    print("fi'(a0) = ", diff_fi(la, a0, func), "\nfi'(b0) = ", diff_fi(la, b0, func))
    flag = True
    cnt_fi = 0
    while True:
        if (abs(diff_fi(la, x, func)) > 1 or (diff_fi(la, a0, func)) > 1 or abs(diff_fi(la, b0, func)) > 1) and flag:
            print("Условие сходимости не выполняется")
            flag = False
        x = fi(func(x), la, x)
        cnt += 1
        cnt_fi += 1
        if abs(func(x)) <= eps or cnt_fi >= 1000:
            return x, func(x), cnt


def plot_function(f, a0, b0):
    x = np.linspace(a0 - 3, b0 + 3, 100)
    y = f(x)
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.xlim(a0 - 3, b0 + 3)
    plt.ylim(min(y) - 3, max(y) + 3)
    plt.show()


def calculate_fx_sys(num, x1, x2):
    if num == 1:
        return [0.3 + 0.1 * np.float64(np.sqrt(np.abs(x1))) ** 2 - 0.2 * np.float64(np.sqrt(np.abs(x2))), 0.7 - 0.2 * np.float64(np.sqrt(np.abs(x1))) ** 2 + 0.1 * x1 * x2]
    elif num == 2:
        return [1.5 - math.cos(x2), (1 + math.sin(x1 - 0.5)) / 2]


def derivative_sys(num, x1, x2):
    if num == 1:
        return [abs(-0.2 * x1) + abs(-0.2), abs(-0.4 * x1 - 0.1 * x2) + abs(-0.1 * x1)]
    elif num == 2:
        return [abs(math.sin(x2)), abs(math.cos((2 * x1 - 1) / 2) / 2)]

def simple_iteration_system(x1_0, x2_0, sys, eps):
    prev_x1, prev_x2 = x1_0, x2_0
    cnt = 0
    flag = True
    cnt_iter = 0
    while True:
        if (max(derivative_sys(sys, x1_0, x2_0)) > 1) and flag:
            print("Условие сходимости не выполняется")
            flag = False
        cnt += 1
        cnt_iter += 1
        x1, x2 = calculate_fx_sys(sys, prev_x1, prev_x2)
        print(x1, x2)
        del_x1 = x1 - prev_x1
        del_x2 = x2 - prev_x2
        # if abs(del_x1) and abs(del_x2) <= eps or cnt_iter >= 10000:
        #     break
        if abs(f1_1(x1, x2)) <= eps and abs(f1_2(x1, x2)) <= eps or abs(f2_1(x1, x2)) <= eps and abs(f2_2(x1, x2)) <= eps or cnt_iter >= 100000:
            break
        prev_x1, prev_x2 = x1, x2
    return x1, x2, cnt, del_x1, del_x2


def plot_functions(f1, f2, x0, y0):
    y_values_1 = np.linspace(x0 - 10, x0 + 10, 400)
    x_values_1 = np.linspace(y0 - 10, y0 + 10, 400)

    z1 = np.zeros((len(x_values_1), len(y_values_1)))

    for i in range(len(x_values_1)):
        for j in range(len(y_values_1)):
            z1[i, j] = f1(y_values_1[j], x_values_1[i])

    x_values_2 = np.linspace(y0 - 10, y0 + 10, 400)
    y_values_2 = np.linspace(x0 - 10, x0 + 10, 400)

    z2 = np.zeros((len(x_values_2), len(y_values_2)))

    for i in range(len(x_values_2)):
        for j in range(len(y_values_2)):
            z2[i, j] = f2(y_values_2[j], x_values_2[i])

    plt.contour(y_values_1, x_values_1, z1, levels=[0], colors='r')
    plt.contour(y_values_2, x_values_2, z2, levels=[0], colors='b')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()


def fi(fn, la, x):
    return x + la * fn


def diff(f, x0):
    h = 1e-6
    return (f(x0 + h) - f(x0)) / h


def diff2(f, x0):
    h = 1e-6
    return (f(x0 + h) - 2*f(x0) + f(x0 - h))/(pow(h, 2))


def diff_fi(la, x, f):
    h = 1e-6
    return (x + h + la * f(x + h) - x - la * f(x)) / h


def f1_1(x1, x2):
    return 0.1 * (x1 ** 2) + x1 + 0.2 * x2 - 0.3


def f1_2(x1, x2):
    return 0.2 * (x1 ** 2) + x2 + 0.1 * x1 * x2 - 0.7


def f2_1(x1, x2):
    return math.cos(x2) + x1 - 1.5


def f2_2(x1, x2):
    return 2 * x2 - math.sin(x1 - 0.5) - 1


print("Выберите ввод данных:")
print("Консоль (1)")
print("Файл (2)")
input_type = int(input())

print("Выберите метод решения:")
print("1. Метод половинного деления")
print("2. Метод Ньютона")
print("3. Метод простой итерации")
print("4. Метод простой итерации для системы")
method = int(input())

if method != 4:
    print("Выберите уравнение для решения:")
    print("1. x^3 - x + 4 = 0")
    print("2. cos(x) = 0")
    print("3. -2,7x^3 -1,48x^2 + 19,23x + 6,35")
    equation = int(input())

    if input_type == 1:
        if equation == 1:
            func = lambda x: x**3 - x + 4
            left = -5
            right = 5
        elif equation == 2:
            func = lambda x: np.cos(x)
            left = -5
            right = 5
        elif equation == 3:
            func = lambda x: -2.7 * x**3 - 1.48 * x**2 + 19.23 * x + 6.35
            left = -8
            right = 8

        simple_graph(func, left, right)

        a0 = float(input("Введите левую границу интервала: "))
        b0 = float(input("Введите правую границу интервала: "))
        eps = float(input("Введите точность: "))

        if not root_exists(func, a0, b0):
            exit()

        if method == 1:
            x, y, iteration = half_division(func, a0, b0, eps)
        elif method == 2:
            x, y, iteration = newton(func, a0, b0, eps)
        elif method == 3:
            x, y, iteration = simple_iteration(func, a0, b0, eps)

        if x is not None:
            print(f"Корень уравнения: x = {x:.10f}")
            print(f"Значение функции в корне: f(x) = {y:.10f}")
            print(f"Число итераций: {iteration}")
            plot_function(func, a0, b0)

    elif input_type == 2:
        filename = input("Введите имя файла: ")
        with open(filename, "r") as file:
            a0 = float(file.readline().strip())
            b0 = float(file.readline().strip())
            eps = float(file.readline().strip())

        if equation == 1:
            func = lambda x: x**3 - x + 4
        elif equation == 2:
            func = lambda x: np.cos(x)
        elif equation == 3:
            func = lambda x: -2.7 * x**3 - 1.48 * x**2 + 19.23 * x + 6.35

        if not root_exists(func, a0, b0):
            exit()

        if method == 1:
            x, y, iteration = half_division(func, a0, b0, eps)
        elif method == 2:
            x, y, iteration = newton(func, a0, b0, eps)
        elif method == 3:
            x, y, iteration = simple_iteration(func, a0, b0, eps)

        output_filename = input("Введите имя файла для вывода: ")
        with open(output_filename, "w") as file:
            if x is not None:
                file.write(f"Корень уравнения: x = {x:.6f}\n")
                file.write(f"Значение функции в корне: y = {y:.6f}\n")
                file.write(f"Число итераций: {iteration}")
            else:
                file.write("На введенном интервале более одного корня.")
        print(f"Результаты успешно записаны в файл {output_filename}")

elif method == 4:
    print("Выберите систему уравнений:")
    print("1. 0.1x1^2 + x1 + 0.2x2 - 0.3 = 0\n 0.2x1^2 + x2 + 0.1x1x2 - 0.7 = 0")
    print("2. cos(x2) + x1 - 1,5 = 0\n 2x2 - sin(x1 - 0,5) - 1 = 0")
    sys = int(input())

    if sys == 1:
        f1, f2 = f1_1, f1_2
        left = -10
        right = 10
    elif sys == 2:
        f1, f2 = f2_1, f2_2
        left = -10
        right = 10

    sys_graph(f1, f2, left, right)

    prev_x1 = int(input("Введите начальное приближение x1_0: "))
    prev_x2 = int(input("Введите начальное приближение x2_0: "))
    e = float(input("Введите точность: "))
    x1, x2, iter_count, d_x1, d_x2 = simple_iteration_system(prev_x1, prev_x2, sys, e)

    print(f"Вектор неизвестных: x1 = {x1}, x2 = {x2}")
    print(f"Значения функции: f1(x1,x2) = {f1(x1, x2)}, f2(x1,x2) = {f2(x1, x2)}")
    print(f"Количество итераций: {iter_count}")
    print(f"Вектор погрешностей: [{d_x1}, {d_x2}]")

    plot_functions(f1, f2, prev_x1, prev_x2)
