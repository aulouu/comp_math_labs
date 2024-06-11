import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
np.seterr(divide='ignore')

def read_data_from_file(file_name):
    with open(file_name, 'r') as file:
        data = [line.split() for line in file.readlines()]
    x = [float(item[0]) for item in data]
    y = [float(item[1]) for item in data]
    return x, y


def input_data_from_console():
    x = []
    y = []
    n = int(input("Введите количество точек (от 8 до 12): "))
    for i in range(n):
        xi, yi = map(float, input(f"Введите координаты точки {i + 1} в формате 'x y': ").split())
        x.append(xi)
        y.append(yi)
    return x, y


def linear_approximation(x, y):
    SX: float = sum([p for p in x])
    SXX: float = sum([p ** 2 for p in x])
    SY: float = sum([p for p in y])
    SXY: float = sum([x[i] * y[i] for i in range(len(x))])
    delta = SXX * len(x) - SX * SX
    delta1 = SXY * len(x) - SX * SY
    delta2 = SXX * SY - SX * SXY

    a = delta1 / delta
    b = delta2 / delta
    func = lambda x: a * x + b
    return func, a, b


def polynomial_approximation_2(x, y):
    SX = sum([p for p in x])
    SX2 = sum([p ** 2 for p in x])
    SX3 = sum([p ** 3 for p in x])
    SX4 = sum([p ** 4 for p in x])
    SY = sum([p for p in y])
    SXY = sum([x[i] * y[i] for i in range(len(x))])
    SX2Y = sum([x[i] * x[i] * y[i] for i in range(len(x))])

    x1 = np.array([[len(x), SX, SX2], [SX, SX2, SX3], [SX2, SX3, SX4]])
    y1 = np.array([SY, SXY, SX2Y])
    a = np.linalg.solve(x1, y1)

    func = lambda x: a[2] * x ** 2 + a[1] * x + a[0]
    return func, a[2], a[1], a[0]


def polynomial_approximation_3(x, y):
    SX = sum([p for p in x])
    SX2 = sum([p ** 2 for p in x])
    SX3 = sum([p ** 3 for p in x])
    SX4 = sum([p ** 4 for p in x])
    SX5 = sum([p ** 5 for p in x])
    SX6 = sum([p ** 6 for p in x])
    SY = sum([p for p in y])
    SXY = sum([x[i] * y[i] for i in range(len(x))])
    SX2Y = sum([x[i] * x[i] * y[i] for i in range(len(x))])
    SX3Y = sum([x[i] * x[i] * x[i] * y[i] for i in range(len(x))])

    x = np.array([[len(x), SX, SX2, SX3], [SX, SX2, SX3, SX4], [SX2, SX3, SX4, SX5], [SX3, SX4, SX5, SX6]])
    y = np.array([SY, SXY, SX2Y, SX3Y])
    a = np.linalg.solve(x, y)

    func = lambda x: a[3] * x ** 3 + a[2] * x ** 2 + a[1] * x + a[0]
    return func, a[3], a[2], a[1], a[0]


def exponential_approximation(x, y):
    if not all([p > 0 for p in x]):
        print("Экспотенциальная аппроксимация невозможна")
        return None, None, None
    y_ln = [np.log(p) for p in y]
    _, b1, a1 = linear_approximation(x, y_ln)
    a = np.exp(a1)
    b = b1
    func = lambda x: a * np.exp(b * x)
    return func, a, b


def logarithmic_approximation(x, y):
    if not all([p > 0 for p in x]):
        print("Логарифмическая аппроксимация невозможна")
        return None, None, None
    x_ln = [np.log(p) for p in x]
    _, a1, b1 = linear_approximation(x_ln, y)
    func = lambda x: a1 * np.log(x) + b1
    return func, a1, b1


def power_approximation(x, y):
    if not (all([p > 0 for p in x]) and all([p > 0 for p in y])):
        print("Степенная аппроксимация невозможна")
        return None, None, None
    x_ln = [np.log(p) for p in x]
    y_ln = [np.log(p) for p in y]
    _, b1, a1 = linear_approximation(x_ln, y_ln)
    a = np.exp(a1)
    b = b1
    func = lambda x: a * x ** b
    return func, a, b


def get_S(f, x, y):
    return sum([(f(x[i]) - y[i]) ** 2 for i in range(len(x))])


def mean_squared_error(x, y, func):
    return np.sqrt(get_S(func, x, y) / len(x))


def pearson_correlation_coefficient(x, y):
    x_ = sum([p for p in x]) / len(x)
    y_ = sum([p for p in y]) / len(y)
    return (sum([(x[i] - x_) * (y[i] - y_) for i in range(len(x))]) /
            np.sqrt(sum([(p - x_) ** 2 for p in x]) * sum([(p - y_) ** 2 for p in y])))


def coefficient_of_determination(x, y, func):
    phi = sum([p for p in x]) / len(x)
    return 1 - (sum([(y[i] - func(x[i])) ** 2 for i in range(len(x))]) / sum([(p - phi) ** 2 for p in y]))


def write_results_to_file(file_name, approximation, coefficients, mse, r_squared):
    with open(file_name, 'w') as file:
        file.write(f'Наилучшая аппроксимирующая функция: {approximation}\n')
        file.write(f'Коэффициенты аппроксимирующей функции: {coefficients}\n')
        file.write(f'Среднеквадратичное отклонение: {mse}\n')
        file.write(f'Коэффициент детерминации: {r_squared}\n')


def main():
    choice = (input("Выберите способ ввода данных (file/console): "))
    if choice == "file":
        file_name = input("Введите имя файла: ")
        x, y = read_data_from_file(file_name)
    else:
        x, y = input_data_from_console()

    f1, a1, b1 = linear_approximation(x, y)
    f2, a2, b2, c2 = polynomial_approximation_2(x, y)
    f3, a3, b3, c3, d3 = polynomial_approximation_3(x, y)
    f4, a4, b4 = exponential_approximation(x, y)
    f5, a5, b5 = logarithmic_approximation(x, y)
    f6, a6, b6 = power_approximation(x, y)

    f = [f1, f2, f3, f4, f5, f6]
    v = ["ax + b", "ax^2 + bx + c", "ax^3 + bx^2 + cx + d", "ae^(bx)", "alnx + b", "ax^b"]
    a = [a1, a2, a3, a4, a5, a6]
    b = [b1, b2, b3, b4, b5, b6]
    c = ["-", c2, c3, "-", "-", "-"]
    d = ["-", "-", d3, "-", "-", "-"]

    titles = ["Линейная аппроксимация", "Квадратичная аппроксимация", "Кубическая аппроксимация",
              "Экспоненциальная аппроксимация", "Логарифмическая аппроксимация", "Степенная аппроксимация"]

    res = PrettyTable()
    res.title = "Выбор аппроксимирующей функции"
    res.field_names = ["Вид функции", "a", "b", "c", "d", "S", "del", "R^2"]
    res.float_format = ".5"

    for i in range(6):
        table = PrettyTable()
        table.title = titles[i]
        table.field_names = ["№", "X", "Y", "P", "eps"]
        table.float_format = ".3"
        if f[i] is None:
            continue
        for j in range(len(x)):
            table.add_row([j + 1, x[j], y[j], f[i](x[j]), f[i](x[j]) - y[j]])
        print("\n", table)
        r2 = coefficient_of_determination(x, y, f[i])
        res.add_row([v[i], a[i], b[i], c[i], d[i], get_S(f[i], x, y), mean_squared_error(x, y, f[i]), r2])
        print(f"Коэффициент детерминации: {r2:.5f}")
        if r2 < 0.5:
            print("Точность аппроксимации недостаточна")
        elif r2 < 0.75:
            print("Слабая аппроксимация")
        elif r2 < 0.95:
            print("Удовлетворительная аппроксимация")
        else:
            print("Высокая точность аппроксимации")

        if i == 0:
            pr = pearson_correlation_coefficient(x, y)
            print(f"Коэффициент Пирсона: {pr:.5f}")
            if pr == 0:
                print("Связь между переменными отсутствует")
            elif pr == 1 or pr == -1:
                print("Строгая линейная зависимость")
            elif pr < 0.3:
                print("Связь слабая")
            elif pr < 0.5:
                print("Связь умеренная")
            elif pr < 0.7:
                print("Связь заметная")
            elif pr < 0.9:
                print("Связь высокая")
            else:
                print("Связь весьма высокая")

    print("\n", res)

    linear_mse = mean_squared_error(x, y, f1)
    polynomial_2_mse = mean_squared_error(x, y, f2)
    polynomial_3_mse = mean_squared_error(x, y, f3)
    if f4 is not None:
        exponential_mse = mean_squared_error(x, y, f4)
    else:
        exponential_mse = np.inf
    if f5 is not None:
        logarithmic_mse = mean_squared_error(x, y, f5)
    else:
        logarithmic_mse = np.inf
    if f6 is not None:
        power_mse = mean_squared_error(x, y, f6)
    else:
        power_mse = np.inf

    approximations = ["Линейная", "Квадратичная", "Кубическая", "Экспоненциальная", "Логарифмическая", "Степенная"]
    mses = [linear_mse, polynomial_2_mse, polynomial_3_mse, exponential_mse, logarithmic_mse, power_mse]
    best_approximation_index = np.argmin(mses)
    print(f"Наилучшая аппроксимирующая функция: {approximations[best_approximation_index]}")

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, label='Исходные данные')
    x_ = np.linspace(min(x) - 1, max(x) + 1, 1000)
    for i in range(6):
        if f[i] is None:
            continue
        yi = np.array([f[i](a) for a in x_])
        plt.plot(x_, yi, label=titles[i])
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
