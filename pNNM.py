import numpy as np
from typing import Callable, List


def create_system(n: int) -> Callable[[np.ndarray], np.ndarray]:

    def system(x: np.ndarray) -> np.ndarray:
        F = np.zeros(n)
        # Первое уравнение
        F[0] = (3 + 2 * x[0]) * x[0] - 2 * x[1] - 3

        # Средние уравнения
        for i in range(1, n - 1):
            F[i] = (3 + 2 * x[i]) * x[i] - x[i - 1] - 2 * x[i + 1] - 2

        # Последнее уравнение
        F[n - 1] = (3 + 2 * x[n - 1]) * x[n - 1] - x[n - 2] - 4

        return F

    return system


def numerical_jacobian(f: Callable[[np.ndarray], np.ndarray],
                       x: np.ndarray,
                       eps: float = 1e-8) -> np.ndarray:
    """Вычисляет матрицу Якоби численным методом"""
    n = len(x)
    J = np.zeros((n, n))
    f_x = f(x)

    for j in range(n):
        x_temp = x.copy()
        x_temp[j] += eps
        J[:, j] = (f(x_temp) - f_x) / eps

    return J


def newton_method(f: Callable[[np.ndarray], np.ndarray],
                  x0: np.ndarray,
                  tol: float = 1e-6,
                  max_iter: int = 100) -> tuple[np.ndarray, int]:
    x = x0.copy()
    for iter in range(max_iter):
        F = f(x)
        if np.linalg.norm(F) < tol:
            return x, iter

        J = numerical_jacobian(f, x)
        try:
            dx = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            raise ValueError("Матрица Якоби вырождена")

        x += dx

    raise ValueError(f"Метод не сошелся за {max_iter} итераций")


def solve_for_different_n(n_values: List[int]):
    for n in n_values:
        print(f"\nРешение для n = {n}:")

        # Создаем систему уравнений
        system = create_system(n)

        # Начальное приближение
        x0 = np.ones(n) * 0.5

        try:
            # Находим решение
            solution, iterations = newton_method(system, x0)

            # Точное решение
            exact_solution = np.ones(n)

            # Вычисляем ошибку
            error = np.linalg.norm(solution - exact_solution)

            print(f"Найденное решение: `{solution}`")
            print(f"Точное решение: {exact_solution}")
            print(f"Ошибка: {error:.2e}")
            print(f"Количество итераций: {iterations}")

        except ValueError as e:
            print(f"Ошибка: {e}")


def callme(n: int):
    #n_values = [3, 5, 10]
    n_values = []
    n_values.append(n)
    solve_for_different_n(n_values)
    
"""
if __name__ == "__main__":
    n_values = [3, 5, 10]
    solve_for_different_n(n_values)
    """