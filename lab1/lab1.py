import numpy as np

def read_data():
    size = 0
    dat = []
    matrix = []
    b = []
    inp_type = input("Выберете способ ввода данных (file/console): ")
    if inp_type == "file":
        filename = input("Введите имя файла: ")
        with open(filename, "r") as f:
            size = int(f.readline())
            for i in range(size):
                matrix.append(list(map(float, f.readline().split(" "))))
    elif inp_type == "console":
        size = int(input("Введите размерность: "))
        print("Введите матрицу:")
        for i in range(size):
            matrix.append(list(map(float, input().split())))
    else:
        print("Некорректный ввод")
        read_data()

    for i in range(size):
        b.append(matrix[i].pop())

    if size > 20 or size <= 1:
        print("Задан некорректный размер")
        read_data()
    dat.append(size)
    dat.append(matrix)
    dat.append(b)
    return dat


class Solver:
    def __init__(self, size, matrix, b):
        self.size = size
        self.matrix = matrix
        self.b = b
        self.x = [0] * size
        self.row_swaps = 0

    def print_matrix(self):
        for i, row in enumerate(self.matrix):
            for element in row:
                print("{:8}".format(element), end='')
            print(" | ", end='')
            print("{:8}".format(self.b[i]))
        print()

    def det_with_libr(self, matrix):
        return np.linalg.det(matrix)

    def det(self, matrix, row_swaps):
        det = -1 ** row_swaps
        for i in range(len(matrix)):
            det *= matrix[i][i]
        # if row_swaps % 2 != 0:
        #     det *= -1
        return det

    def upper_triangular(self):
        for i in range(self.size):
            if self.matrix[i][i] == 0:
                non_zero_row = i + 1
                while non_zero_row < self.size and self.matrix[non_zero_row][i] == 0:
                    non_zero_row += 1

                if non_zero_row == self.size:
                    print("Система несовместна.")
                    exit()

                self.matrix[i], self.matrix[non_zero_row] = self.matrix[non_zero_row], self.matrix[i]
                self.b[i], self.b[non_zero_row] = self.b[non_zero_row], self.b[i]
                self.row_swaps += 1
                # print(f"Матрица после перестановки {self.row_swaps}:")
                # self.print_matrix()

            for j in range(i + 1, self.size):
                coef = self.matrix[j][i] / self.matrix[i][i]
                for k in range(self.size):
                    self.matrix[j][k] -= coef * self.matrix[i][k]
                self.b[j] -= coef * self.b[i]
            # print(f"Матрица после итерации {i + 1}:")
            # self.print_matrix()

    def back_substitution(self):
        for i in range(self.size - 1, -1, -1):
            self.x[i] = self.b[i]
            for j in range(i + 1, self.size):
                self.x[i] -= self.matrix[i][j] * self.x[j]
            self.x[i] /= self.matrix[i][i]

    def compute_residuals(self):
        residuals = [0] * len(self.b)
        for i in range(len(self.b)):
            sum_val = 0
            for j in range(len(self.x)):
                sum_val += self.matrix[i][j] * self.x[j]
            residuals[i] = sum_val - self.b[i]
        return residuals

if __name__ == '__main__':
    data = read_data()
    size, matrix, b = data[0], data[1], data[2]
    solver = Solver(size, matrix, b)

    print("Получена матрица:")
    solver.print_matrix()

    det_with_libr = solver.det_with_libr(matrix)
    print("Определитль матрицы, посчитанный библиотекой:")
    print(det_with_libr)
    print()

    solver.upper_triangular()
    print("Верхнетреугольная матрица:")
    solver.print_matrix()
    print("Количество перестановок строк матрицы:", solver.row_swaps)
    print()

    det = solver.det(matrix, solver.row_swaps)
    print("Определитель матрицы:")
    print(det)
    print()

    solver.back_substitution()
    print("Вектор неизвестных:")
    for i in range(size):
        print(f"x{i + 1} = {solver.x[i]}")
    print()

    residuals = solver.compute_residuals()
    print("Вектор невязок:")
    for i in range(size):
        print('{0:.17e}'.format(residuals[i]))
    print()
