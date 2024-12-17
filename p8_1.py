import numpy as np
from math import pi


def kernel(x, s):
    """Ядро интегрального уравнения"""
    return 1 / (pi * (1 + (x - s) ** 2))


def free_term(x):
    """Свободный член уравнения"""
    return 1


def solve_fredholm(n):
    """
    Решение интегрального уравнения методом квадратур
    n - количество отрезков разбиения
    """
    # Шаг интегрирования
    h = 2 / n

    # Узлы квадратурной формулы
    x = np.linspace(-1, 1, n + 1)

    # Формируем матрицу системы
    A = np.zeros((n + 1, n + 1))
    b = np.zeros(n + 1)

    # Заполняем матрицу и вектор свободных членов
    for i in range(n + 1):
        for j in range(n + 1):
            if j == 0 or j == n:
                A[i, j] = kernel(x[i], x[j]) * h / 2
            else:
                A[i, j] = kernel(x[i], x[j]) * h

        A[i, i] -= 1  # вычитаем единичную матрицу
        b[i] = -free_term(x[i])  # знак минус из-за переноса в правую часть

    # Решаем систему линейных уравнений
    y = np.linalg.solve(A, b)

    return x, -y  # возвращаем знак решения


def main():
    try:
        n = int(input("Введите количество отрезков разбиения: "))

        x, y = solve_fredholm(n)

        print("\nПриближенное решение:")
        print("   x      y(x)")
        print("-" * 20)
        for i in range(len(x)):
            print(f"{x[i]:7.3f} {y[i]:10.6f}")


    except ValueError:
        print("Ошибка! Введите целое число.")
    except np.linalg.LinAlgError:
        print("Ошибка! Не удалось решить систему уравнений.")


if __name__ == "__main__":
    main()