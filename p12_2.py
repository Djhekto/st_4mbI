import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import streamlit as st


class ParabolicSolver2D:
    def __init__(self, l1, l2, T, N1, N2, M):
        """
        Инициализация параметров задачи
        l1, l2 - размеры прямоугольника
        T - конечное время
        N1, N2 - количество точек по пространству
        M - количество точек по времени
        """
        self.l1 = l1
        self.l2 = l2
        self.T = T
        self.N1 = N1
        self.N2 = N2
        self.M = M

        # Вычисление шагов сетки
        self.h1 = l1 / N1
        self.h2 = l2 / N2
        self.tau = T / M

        # Создание сеток
        self.x1 = np.linspace(0, l1, N1 + 1)
        self.x2 = np.linspace(0, l2, N2 + 1)
        self.t = np.linspace(0, T, M + 1)

        # Создание массива решения
        self.u = np.zeros((M + 1, N1 + 1, N2 + 1))

    def exact_solution(self, x1, x2, t):
        """Точное решение задачи"""
        return t * x1 * (self.l1 - x1) * x2 * (self.l2 - x2)

    def initial_condition(self, x1, x2):
        """Начальное условие"""
        return 0.0

    def f(self, x1, x2, t):
        """Правая часть уравнения"""
        return (x1 * (self.l1 - x1) * x2 * (self.l2 - x2) +
                t * (-2 * x2 * (self.l2 - x2) - 2 * x1 * (self.l1 - x1)))

    def solve(self):
        """Решение задачи методом переменных направлений"""
        # Заполнение начального условия
        for i in range(self.N1 + 1):
            for j in range(self.N2 + 1):
                self.u[0, i, j] = self.initial_condition(self.x1[i], self.x2[j])

        # Коэффициенты для прогонки
        r1 = self.tau / (2 * self.h1 ** 2)
        r2 = self.tau / (2 * self.h2 ** 2)

        # Матрицы прогонки для каждого направления
        A1 = np.zeros((self.N1 - 1, self.N1 - 1))
        A2 = np.zeros((self.N2 - 1, self.N2 - 1))

        # Заполнение матриц прогонки
        for i in range(self.N1 - 1):
            if i > 0:
                A1[i, i - 1] = -r1
            A1[i, i] = 1 + 2 * r1
            if i < self.N1 - 2:
                A1[i, i + 1] = -r1

        for i in range(self.N2 - 1):
            if i > 0:
                A2[i, i - 1] = -r2
            A2[i, i] = 1 + 2 * r2
            if i < self.N2 - 2:
                A2[i, i + 1] = -r2

        # Решение по временным слоям
        for n in range(self.M):
            # Промежуточный слой (прогонка по x1)
            u_half = np.zeros((self.N1 + 1, self.N2 + 1))

            for j in range(1, self.N2):
                # Формирование правой части для прогонки по x1
                b = np.zeros(self.N1 - 1)
                for i in range(1, self.N1):
                    b[i - 1] = (self.u[n, i, j] +
                                r2 * (self.u[n, i, j + 1] - 2 * self.u[n, i, j] + self.u[n, i, j - 1]) +
                                0.5 * self.tau * self.f(self.x1[i], self.x2[j], self.t[n]))

                # Решение СЛАУ для внутренних точек
                u_half[1:self.N1, j] = np.linalg.solve(A1, b)

            # Новый временной слой (прогонка по x2)
            for i in range(1, self.N1):
                # Формирование правой части для прогонки по x2
                b = np.zeros(self.N2 - 1)
                for j in range(1, self.N2):
                    b[j - 1] = (u_half[i, j] +
                                r1 * (u_half[i + 1, j] - 2 * u_half[i, j] + u_half[i - 1, j]) +
                                0.5 * self.tau * self.f(self.x1[i], self.x2[j], self.t[n + 1]))

                # Решение СЛАУ для внутренних точек
                self.u[n + 1, i, 1:self.N2] = np.linalg.solve(A2, b)

    def calculate_error(self):
        """Вычисление максимальной абсолютной погрешности"""
        error = 0.0
        for n in range(self.M + 1):
            for i in range(self.N1 + 1):
                for j in range(self.N2 + 1):
                    exact = self.exact_solution(self.x1[i], self.x2[j], self.t[n])
                    error = max(error, abs(self.u[n, i, j] - exact))
        return error

    def plot_solution(self, time_index):
        """Построение графика решения в заданный момент времени"""
        X1, X2 = np.meshgrid(self.x1, self.x2)

        # Численное решение
        fig = plt.figure(figsize=(12, 5))

        # График численного решения
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot_surface(X1, X2, self.u[time_index].T, cmap='viridis')
        ax1.set_xlabel('x1')
        ax1.set_ylabel('x2')
        ax1.set_zlabel('u')
        ax1.set_title(f'Численное решение (t = {self.t[time_index]:.3f})')

        # График точного решения
        ax2 = fig.add_subplot(122, projection='3d')
        U_exact = np.zeros((self.N1 + 1, self.N2 + 1))
        for i in range(self.N1 + 1):
            for j in range(self.N2 + 1):
                U_exact[i, j] = self.exact_solution(self.x1[i], self.x2[j], self.t[time_index])

        ax2.plot_surface(X1, X2, U_exact.T, cmap='viridis')
        ax2.set_xlabel('x1')
        ax2.set_ylabel('x2')
        ax2.set_zlabel('u')
        ax2.set_title(f'Точное решение (t = {self.t[time_index]:.3f})')

        plt.tight_layout()
        #plt.show()
        st.pyplot(ax2)
        st.pyplot(ax1)


def main(N=20, M= 20):
    # Параметры задачи
    l1 = l2 = 1.0  # размеры прямоугольника
    T = 1.0  # конечное время

    try:
        print("\nВведите параметры сетки:")
        #N = int(input("Количество точек по пространству (N): "))
        #M = int(input("Количество точек по времени (M): "))

        # Создание и решение задачи
        solver = ParabolicSolver2D(l1, l2, T, N, N, M)
        solver.solve()

        # Вычисление погрешности
        error = solver.calculate_error()
        print(f"\nМаксимальная погрешность: {error:.2e}")

        # Построение графиков в различные моменты времени
        times = [0, M // 2, M]  # начало, середина и конец временного интервала
        for t in times:
            solver.plot_solution(t)


    except ValueError:
        print("Ошибка ввода! Проверьте формат данных.")
    except np.linalg.LinAlgError:
        print("Ошибка! Не удалось решить систему уравнений.")
    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")


if __name__ == "__main__":
    main()