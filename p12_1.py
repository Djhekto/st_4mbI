import numpy as np
import matplotlib.pyplot as plt
from math import pi, exp, sin


def k(x):
    """Коэффициент теплопроводности"""
    return 1.0


def f(x, t):
    """Правая часть уравнения"""
    return (-pi ** 2 * exp(-pi ** 2 * t) * sin(pi * x) +
            -16 * pi ** 2 * exp(-16 * pi ** 2 * t) * sin(4 * pi * x))


def v(x):
    """Начальное условие"""
    return sin(pi * x) + sin(4 * pi * x)


def exact_solution(x, t):
    """Точное решение"""
    return exp(-pi ** 2 * t) * sin(pi * x) + exp(-16 * pi ** 2 * t) * sin(4 * pi * x)


def solve_parabolic(N, M, T, L, q):
    """
    Решение параболической задачи
    N - число точек по пространству
    M - число точек по времени
    T - конечное время
    L - длина отрезка
    q - вес схемы
    """
    # Шаги сетки
    h = L / N  # по пространству
    tau = T / M  # по времени

    # Создание сеток
    x = np.linspace(0, L, N + 1)
    t = np.linspace(0, T, M + 1)

    # Создание массива решения
    u = np.zeros((M + 1, N + 1))

    # Начальное условие
    for i in range(N + 1):
        u[0, i] = v(x[i])

    # Граничные условия
    for j in range(M + 1):
        u[j, 0] = 0
        u[j, N] = 0

    # Коэффициенты для матрицы системы
    r = tau / (h ** 2)

    # Формирование матрицы системы
    A = np.zeros((N - 1, N - 1))
    for i in range(N - 1):
        if i > 0:
            A[i, i - 1] = -q * r
        A[i, i] = 1 + 2 * q * r
        if i < N - 2:
            A[i, i + 1] = -q * r

    # Решение по временным слоям
    for j in range(M):
        # Формирование правой части
        b = np.zeros(N - 1)
        for i in range(N - 1):
            b[i] = u[j, i + 1] + tau * ((1 - q) * f(x[i + 1], t[j]) + q * f(x[i + 1], t[j + 1]))
            if i > 0:
                b[i] += (1 - q) * r * u[j, i]
            if i < N - 2:
                b[i] += (1 - q) * r * u[j, i + 2]
            b[i] += (1 - q) * r * (-2 * u[j, i + 1])

        # Решение СЛАУ
        u[j + 1, 1:N] = np.linalg.solve(A, b)

    return x, t, u


def calculate_error(u_num, u_exact, x, t):
    """Вычисление максимальной абсолютной погрешности"""
    error = 0.0
    for i in range(len(t)):
        for j in range(len(x)):
            error = max(error, abs(u_num[i, j] - u_exact(x[j], t[i])))
    return error


def plot_results(x, t, u_num, title):
    """Построение графика численного решения"""
    X, T = np.meshgrid(x, t)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, T, u_num, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('u')
    ax.set_title(title)
    plt.show()


def main():
    # Параметры задачи
    L = 1.0  # длина отрезка
    T = 0.1  # конечное время

    # Списки параметров для экспериментов
    N_values = [20, 40, 80]  # точки по пространству
    M_values = [20, 40, 80]  # точки по времени
    q_values = [0.5, 1.0]  # веса схемы

    for q in q_values:
        print(f"\nВес схемы q = {q}")
        print("\nТаблица погрешностей:")
        print("   N    M      Погрешность")
        print("-" * 30)

        for N in N_values:
            for M in M_values:
                # Решение задачи
                x, t, u_num = solve_parabolic(N, M, T, L, q)

                # Вычисление погрешности
                error = calculate_error(u_num, exact_solution, x, t)

                print(f"{N:4d} {M:4d} {error:15.2e}")

                # Построение графика для некоторых параметров
                if N == 40 and M == 40:
                    title = f"Численное решение (q={q}, N={N}, M={M})"
                    plot_results(x, t, u_num, title)

                    # Построение точного решения для сравнения
                    u_exact = np.zeros((M + 1, N + 1))
                    for i in range(M + 1):
                        for j in range(N + 1):
                            u_exact[i, j] = exact_solution(x[j], t[i])
                    plot_results(x, t, u_exact, f"Точное решение (N={N}, M={M})")


if __name__ == "__main__":
    main()