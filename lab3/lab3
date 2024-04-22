import math
import scipy.integrate as spi

def f1(x):
    return math.cos(x)

def f2(x):
    return math.exp(x)

def f3(x):
    return x**2

def f4(x):
    return -2 * x**3 - 3 * x**2 + x + 5


def left_rectangles(func, a, b, n):
    h = (b - a) / n
    x = a
    integral = 0
    for i in range(n):
        integral += func(x)
        x += h
    return integral * h


def right_rectangles(func, a, b, n):
    h = (b - a) / n
    x = a + h
    integral = 0
    for i in range(n):
        integral += func(x)
        x += h
    return integral * h


def mid_rectangles(func, a, b, n):
    h = (b - a) / n
    x = a + h / 2
    integral = 0
    for i in range(n):
        integral += func(x)
        x += h
    return integral * h


def trapezoids(func, a, b, n):
    h = (b - a) / n
    x = a
    integral = 0
    integral += func(x) / 2
    x += h
    for i in range(n - 1):
        integral += func(x)
        x += h
    # print(x)
    integral += func(x) / 2
    return integral * h


def simpson(func, a, b, n):
    h = (b - a) / n
    x = a
    integral = 0
    integral += func(x)
    x += h
    for i in range(1, n, 2):
        integral += 4 * func(x)
        x += 2 * h
    x = a + 2 * h
    for i in range(2, n - 1, 2):
        integral += 2 * func(x)
        x += 2 * h
    integral += func(b)
    return integral * h / 3


def calculate_integral(method, func, a, b, eps, k):
    n = 4
    I1 = method(func, a, b, n)
    n *= 2
    I2 = method(func, a, b, n)
    print(n, I2, abs(I1 - I2) / (2 ** k - 1))
    if abs(I1 - I2) / (2 ** k - 1) < eps:
        return I1, n
    while abs(I1 - I2) / (2 ** k - 1) >= eps:
        n *= 2
        I1 = I2
        I2 = method(func, a, b, n)
        print(n, I2, abs(I1 - I2) / (2 ** k - 1))
    # exact_integral = spi.quad(func, a, b)[0]
    # absolute_error = abs(exact_integral - I2)
    # relative_error = absolute_error / abs(exact_integral) * 100
    return I2, n #, exact_integral, absolute_error, relative_error


def main():
    print("Выберите функцию:")
    print("1. cos(x)")
    print("2. e^x")
    print("3. x^2")
    print("4. -2x^3 - 3x^2 + x + 5")
    func_num = int(input())

    if func_num < 1 or func_num > 4:
        print("Неверный номер функции.")
        exit()

    func_list = [f1, f2, f3, f4]
    func = func_list[func_num - 1]

    a = float(input("Введите левый предел интегрирования: "))
    b = float(input("Введите правый предел интегрирования: "))
    eps = float(input("Введите требуемую точность: "))

    print("Выберите метод:")
    print("1. Метод прямоугольников (левые)")
    print("2. Метод прямоугольников (правые)")
    print("3. Метод прямоугольников (средние)")
    print("4. Метод трапеций")
    print("5. Метод Симпсона")
    method_num = int(input())

    method_list = [left_rectangles, right_rectangles, mid_rectangles, trapezoids, simpson]
    method = method_list[method_num - 1]
    k_values = {1: 1, 2: 1, 3: 2, 4: 2, 5: 4}

    integral, n = calculate_integral(method, func, a, b, eps, k_values[method_num])
    # print(f"Точное значение интеграла (с помощью библиотеки): {exact_integral}")
    print(f"Значение интеграла: {integral}")
    print(f"Число разбиений интервала интегрирования: {n}")
    # print(f"Абсолютная погрешность: {absolute_error}")
    # print(f"Относительная погрешность: {relative_error}%")


if __name__ == "__main__":
    main()
