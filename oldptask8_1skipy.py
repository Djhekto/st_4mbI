import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import trapezoid, simpson

# Ядро уравнения
def kernel(x, s):
    return 1 / (np.pi * (1 + (x - s)**2))

# Свободный член
def rhs(x):
    return 1

# -----------------------------
# Метод квадратур (точечная формула) для решения Фредгольма 2-го рода
# -----------------------------
def fredholm_second_kind(a, b, n, kernel, rhs, method='trapezoid'):
    """
    "Кастомный" метод: дискретная квадратура по узлам [a,b].
    """
    x = np.linspace(a, b, n)
    h = (b - a) / (n - 1)
    A = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            weight = h
            if method == 'trapezoid':
                if j == 0 or j == n - 1:
                    weight /= 2
            elif method == 'simpson':
                if n % 2 == 0:
                    raise ValueError("Метод Симпсона требует нечетного числа узлов")
                if j == 0 or j == n - 1:
                    weight /= 3
                elif j % 2 == 0:
                    weight *= 2/3
                else:
                    weight *= 4/3
            elif method == 'rectangle':
                pass  # Прямоугольный метод: один вес h у каждого узла
            else:
                raise ValueError(f"Unknown quadrature method: {method}")

            A[i, j] = (1 if i == j else 0) - weight * kernel(x[i], x[j])

    b_vec = np.array([rhs(xi) for xi in x])
    y = np.linalg.solve(A, b_vec)

    return x, y

# -----------------------------
# Базис Лагранжа для коллокации
# -----------------------------
def lagrange_basis(x_nodes, j, s):
    """
    Полином Лагранжа L_j(s) на сетке x_nodes: L_j(x_j)=1, L_j(x_m)=0 (m!=j).
    s может быть массивом (для векторной оценки).
    """
    L = np.ones_like(s, dtype=float)
    xj = x_nodes[j]
    for m, xm in enumerate(x_nodes):
        if m != j:
            L *= (s - xm) / (xj - xm)
    return L

# -----------------------------
# Коллокационный метод через scipy-интегрирование
# -----------------------------
def fredholm_second_kind_scipy(f, k, a, b, n, method='trapezoid'):
    """
    Коллокация с базисом Лагранжа и библиотечным интегрированием (scipy).
    A_{ij} = δ_{ij} - \int_a^b k(x_i, s)*L_j(s) ds
    """
    x_nodes = np.linspace(a, b, n)
    A = np.eye(n)
    f_values = np.array([f(xi) for xi in x_nodes])

    # Сетка для интегрирования (нам нужна более частая сетка, чтобы точно взять интеграл)
    s_mesh = np.linspace(a, b, 10*n)
    dx = s_mesh[1] - s_mesh[0]  # Вычисляем шаг

    for i in range(n):
        for j in range(n):
            # g_ij(s) = k(x_nodes[i], s) * L_j(s)
            L_j_s = lagrange_basis(x_nodes, j, s_mesh)
            integrand = k(x_nodes[i], s_mesh) * L_j_s

            if method == 'trapezoid':
                integral_val = trapezoid(integrand, s_mesh)
            elif method == 'simpson':
                # Используем dx вместо x, так как в старых версиях SciPy simpson не принимает x
                integral_val = simpson(integrand, dx=dx)
            elif method == 'rectangle':
                # Реализуем "прямоугольный" метод на сетке s_mesh
                integral_val = np.sum(integrand) * dx
            else:
                raise ValueError("Неизвестный метод!")

            A[i, j] -= integral_val

    y = np.linalg.solve(A, f_values)
    return x_nodes, y

# -----------------------------
# Streamlit-приложение
# -----------------------------
def run_task8_1():
    st.title("Решение интегрального уравнения Фредгольма второго рода")

    st.markdown(""" 
    **Методы решения:**
    1. **Без библиотеки Scipy** — метод квадратур.
    2. **С использованием библиотеки Scipy** — Коллокационный метод с численным интегрированием через `scipy.integrate`,  коллокация с базисом Лагранжа.

    $A_{ij} = \delta_{ij} - \int_a^b k(x_i, s) L_j(s) \, ds$.
    """)


    # Ввод параметров
    a = st.number_input("Левая граница отрезка (a):", value=-1.0)
    b = st.number_input("Правая граница отрезка (b):", value=1.0)
    n = st.number_input("Количество узлов (n):", min_value=3, max_value=50, value=3)
    method = st.selectbox("Выберите метод квадратуры:", ["trapezoid", "rectangle", "simpson"])

    if st.button("Рассчитать"):
        try:
            # Решение "кастомным" методом
            x, y = fredholm_second_kind(a, b, n, kernel, rhs, method)

            # Решение коллокацией с scipy-интегрированием
            x_scipy, y_scipy = fredholm_second_kind_scipy(rhs, kernel, a, b, n, method)

            # Построение графика
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(x, y, label=f'$Без\;Scipy\;-\;метод\;{method}$', marker='o')
            ax.plot(x_scipy, y_scipy, linestyle='--', label=f'$С\;Scipy\;-\;метод\;{method}$', marker='x')
            ax.set_xlabel("$x$")
            ax.set_ylabel("$y(x)$")
            ax.set_title("$Решение\;интегрального\;уравнения\;Фредгольма\;второго\;рода$")
            ax.legend()
            ax.grid(True)

            st.pyplot(fig)

            # Вывод результатов в виде таблицы
            results = pd.DataFrame({
                "x": x,
                "y (без Scipy)": y,
                "y (с Scipy)": y_scipy
            })
            st.dataframe(results)

        except Exception as e:
            st.error(f"Произошла ошибка: {e}")

# Запуск приложения
if __name__ == "__main__":
    run_task8_1()
