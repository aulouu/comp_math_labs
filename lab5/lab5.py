import math
from math import factorial
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable


def input_data_from_console():
    x = []
    y = []
    n = int(input("Введите количество точек: "))
    for i in range(n):
        xi, yi = map(float, input(f"Введите координаты точки {i + 1} в формате 'x y': ").split())
        x.append(xi)
        y.append(yi)
    return x, y

def read_data_from_file(file_name):
    with open(file_name, 'r') as file:
        data = [line.split() for line in file.readlines()]
    x = [float(item[0]) for item in data]
    y = [float(item[1]) for item in data]
    return x, y

def generate_data(func, a, b, n):
    x = np.linspace(a, b, n)
    y = func(x)
    return x, y

def check_equidistant(x):
    h = x[1] - x[0]
    for i in range(len(x) - 1):
        if not math.isclose(x[i + 1] - x[i], h):
            return False
    return True

def differences(x, y):
    n = len(x)
    table = [[0] * n for _ in range(n)]
    for i in range(n):
        table[i][0] = y[i]
    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = table[i + 1][j - 1] - table[i][j - 1]
    return table

def print_differences_table(x, y):
    differences_table = differences(x, y)
    n = len(x)
    table = PrettyTable()
    table.title = "Таблица конечных разностей"
    headers = ['x'] + [f'd^{i}y' for i in range(n)]
    table.field_names = headers
    for i in range(n):
        row = [f'{x[i]:.4f}']
        for j in range(n):
            if differences_table[i][j] == 0:
                row.append('-')
            else:
                row.append(f'{differences_table[i][j]:.4f}')
        table.add_row(row)
    print(table)

def lagrang_interpolation(x, y, xi):
    result = 0
    for i in range(len(y)):
        li = 1
        for j in range(len(x)):
            li *= (xi - x[j]) / (x[i] - x[j]) if i != j else 1
        result += y[i] * li
    return result

def divided_differences(x, y):
    n = len(x)
    table = [[0] * n for _ in range(n)]
    for i in range(n):
        table[i][0] = y[i]
    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = (table[i + 1][j - 1] - table[i][j - 1]) / (x[i + j] - x[i])
    return table

def newton_interpolation(x, y, xi):
    a = divided_differences(x, y)[0]
    result = a[0]
    for i in range(1, len(x)):
        term = a[i]
        for j in range(i):
            term *= (xi - x[j])
        result += term
    return result

def gauss_t(t, n):
    tmp = t
    for i in range(1, n):
        tmp *= t + ((-1) ** i) * ((i + 1) // 2)
    return tmp

def gauss_tt(t, n):
    tmp = t
    for i in range(1, n):
        tmp *= t - ((-1) ** i) * ((i + 1) // 2)
    return tmp

def gauss_interpolation(x, y, xi):
    n = len(x)
    y = differences(x, y)
    mid = n // 2
    h = x[1] - x[0]
    if xi >= x[mid]:
        t = (xi - x[mid]) / h
        result = y[mid][0]
        for i in range(1, n):
            index = mid - i // 2
            if index < 0:
                break
            result += gauss_t(t, i) * y[index][i] / factorial(i)
    else:
        t = (xi - x[mid]) / h
        result = y[mid][0]
        for i in range(1, n):
            if i % 2 == 1:
                index = mid - (i // 2)
            else:
                index = mid - (i // 2) + 1
            if index < 0:
                break
            result += gauss_tt(t, i) * y[index - 1][i] / factorial(i)
    return result

def plot_function_and_interpolation(x, y, xi, yi_gauss, func):
    plt.figure(figsize=(10, 6))

    x_func = np.linspace(min(x), max(x), 100)
    y_func = func(x_func)
    plt.plot(x_func, y_func, label='Исходная функция')

    plt.scatter(x, y, c='r', label='Узлы интерполяции')

    x_gauss = np.linspace(min(x), max(x), 100)
    y_gauss = [gauss_interpolation(x, y, x_) for x_ in x_gauss]
    plt.plot(x_gauss, y_gauss, c='g', label='Интерполяционный многочлен Гаусса')

    plt.scatter(xi, yi_gauss, c='g', label='Точка интерполяции (Гаусс)')
    plt.xlim(min(x) - 0.5, max(x) + 0.5)
    plt.ylim(min(y_func) - 0.5, max(y_func) + 0.5)
    plt.grid()
    plt.legend()
    plt.show()

def plot_interpolation(x, y, xi, yi_gauss):
    plt.figure(figsize=(10, 6))

    plt.scatter(x, y, c='r', label='Узлы интерполяции')

    x_gauss = np.linspace(min(x), max(x), 100)
    y_gauss = [gauss_interpolation(x, y, x_) for x_ in x_gauss]
    plt.plot(x_gauss, y_gauss, c='g', label='Интерполяционный многочлен Гаусса')

    plt.scatter(xi, yi_gauss, c='g', label='Точка интерполяции (Гаусс)')
    plt.xlim(min(x) - 0.5, max(x) + 0.5)
    plt.ylim(min(y) - 0.5, max(y) + 0.5)
    plt.grid()
    plt.legend()
    plt.show()

def main():
    func = False
    print("Выберите способ ввода данных:")
    print("1. Ввод с клавиатуры")
    print("2. Ввод из файла")
    print("3. На основе выбранной функции")
    choice = int(input())
    if choice == 1:
        x, y = input_data_from_console()
    elif choice == 2:
        file_name = input("Введите имя файла: ")
        x, y = read_data_from_file(file_name)
    else:
        print("Выберите функцию:")
        print("1. cos(x)")
        print("2. 2^x")
        print("3. x^7 - 3x^4 + x^3 - 5x")
        func_num = int(input())
        if func_num == 1:
            func = lambda t: np.cos(t)
        elif func_num == 2:
            func = lambda t: np.power(2, t)
        else: func = lambda t: t ** 7 - 3 * (t ** 4) + t ** 3 - 5*t

        a = float(input("Введите левую границу интервала: "))
        b = float(input("Введите правую границу интервала: "))
        n = int(input("Введите количество точек: "))

        x, y = generate_data(func, a, b, n)

    inp_table = PrettyTable()
    inp_table.title = "Исходные данные"
    inp_table.field_names = ["x", "y"]
    inp_table.float_format = ".4"
    for i in range(len(x)):
        inp_table.add_row([x[i], y[i]])
    print(inp_table)

    xi = float(input("Введите значение аргумента для интерполяции: "))
    if check_equidistant(x):
        print_differences_table(x, y)

    yi_lagr = lagrang_interpolation(x, y, xi)
    yi_newt = newton_interpolation(x, y, xi)
    yi_gauss = gauss_interpolation(x, y, xi)
    if not check_equidistant(x):
        print(f"Приближенное значение функции по многочлену Лагранжа: {yi_lagr:.4f}")
        print(f"Приближенное значение функции по многочлену Ньютона с разделенными разностями: {yi_newt:.4f}")
        print("Узлы не являются равноотстоящими, метод Гаусса не применим.")
    else:
        print(f"Приближенное значение функции по многочлену Лагранжа: {yi_lagr:.4f}")
        print(f"Узлы являются равноотстоящими, метод Ньютона с разделенными разностями не применим.")
        print(f"Приближенное значение функции по многочлену Гаусса: {yi_gauss:.4f}")
        if func != False:
            plot_function_and_interpolation(x, y, xi, yi_gauss, func)
        else:
            plot_interpolation(x, y, xi, yi_gauss)


if __name__ == '__main__':
    main()
