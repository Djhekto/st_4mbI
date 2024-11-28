import math
from typing import Callable
def bisection_method(f: Callable[[float], float], a: float, b: float, epsilon: float = 1e-6, max_iter: int = 1000) -> tuple[float, int]:
    if f(a) * f(b) > 0:
        raise ValueError("Функция должна иметь разные знаки на концах интервала")

    iteration = 0
    while (b - a) > epsilon and iteration < max_iter:
        c = (a + b) / 2  # середина интервала
        if abs(f(c)) < epsilon:  # если нашли корень с нужной точностью
            return c, iteration
        if f(a) * f(c) < 0:  # если корень в левой половине
            b = c
        else:  # если корень в правой половине
            a = c
        iteration += 1

    return (a + b) / 2, iteration


def equation(x: float) -> float:
    return (1 + x ** 2) * math.exp(-x) + math.sin(x)


def find_intervals(f: Callable[[float], float], a: float, b: float, steps: int = 1000) -> list[tuple[float, float]]:
    intervals = []
    dx = (b - a) / steps
    x_prev = a
    y_prev = f(x_prev)

    for i in range(1, steps + 1):
        x_curr = a + i * dx
        y_curr = f(x_curr)
        if y_prev * y_curr <= 0:
            intervals.append((x_prev, x_curr))
        x_prev, y_prev = x_curr, y_curr

    return intervals

"""
def find_intervals(f, a, b, steps=1000):
    intervals = []
    dx = (b - a) / steps
    x_prev = a
    y_prev = f(x_prev)

    for i in range(1, steps + 1):
        x_curr = a + i * dx
        y_curr = f(x_curr)
        if y_prev * y_curr <= 0:
            intervals.append((x_prev, x_curr))
        x_prev, y_prev = x_curr, y_curr

    return intervals
"""

def main():
    intervals = find_intervals(equation, 0, 10)

    print("Найденные корни уравнения:")
    print("(1+x^2)e^(-x)+sin(x)=0")
    print("\nx         f(x)              Итераций")
    print("-" * 45)

    for a, b in intervals:
        try:
            root, iterations = bisection_method(equation, a, b)
            print(f"{root:.6f}  {equation(root):15.2e}  {iterations:8d}")
        except ValueError as e:
            print(f"Ошибка на интервале [{a}, {b}]: {e}")


if __name__ == "__main__":
    main()