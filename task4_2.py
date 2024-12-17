import streamlit as st
import numpy as np
import io
from contextlib import redirect_stdout
import matplotlib.pyplot as plt
import re

import pNNM
import p4_2
import pypy


def evaluate_single_fragment(s: str) -> str:
    """
    Find and evaluate a single fragment between backticks in a string.
    
    Args:
    s (str): The input string containing a fragment between backticks.
    
    Returns:
    str: The evaluated result of the first occurrence of a backticked fragment.
    """
    start = s.find('`')
    if start == -1:
        return ""
    
    end = s.find('`', start + 1)
    if end == -1:
        return ""
    
    fragment = s[start + 1:end]
    try:
        evaluated = eval(fragment)
        return re.sub('`', '', str(evaluated))
    except Exception as e:
        return f"{fragment}"
    

def run_task4_2():
    
    menu = st.sidebar.radio("Выберете страницу:",
        (   "Постановка задачи",
            "Теория",
            "Про метод решения",
            "Про решение SciPy",
            "Демонстрация работы",
            "Дополнительная информация",
        )
    )

    if menu == "Теория":
        display_theory()
    elif menu == "Постановка задачи":
        display_task_info()
    elif menu == "Про метод решения":
        display_theory_noscipy()
    elif menu == "Про решение SciPy":
        display_theory_scipy()
    elif menu == "Демонстрация работы":
        display_task_working()
    elif menu == "Дополнительная информация":
        display_extra()
    

def display_task_info():
    st.subheader("Задание 4.2")
    st.write("""
    ##### 4 Нелинейные уравнения и системы
    
    **Задание 4.2**

    Напишите программу для нахождения решения системы нелинейных уравнений
    $F(x) = 0$ методом Ньютона при численном вычислении матрицы Якоби.

    С ее помощью найдите приближенное решение системы

    $\\begin{aligned}
    (3 + 2x_1) x_1 - 2 x_2 = 3,
    \end{aligned}$

    $\\begin{aligned}
    (3 + 2x_i) x_i - x_{i-1} - 2 x_{i+1} = 2,
    \quad i = 2, 3, ..., n-1,
    \end{aligned}$

    $\\begin{aligned}
    (3 + 2x_n) x_n - x_{n-1} = 4
    \end{aligned}$

    и сравните его с точным решением
    $x_i = 1, \ i = 1,2,..., n$
    при различных $n$.

    Решите также эту задачу с помощью библиотеки SciPy.
    """)

def display_theory():
    #st.subheader("Метод Ньютона при вычислении матрицы Якоби для решения системы нелинейных уравнений ")
    # Implement theory content for Task 1
    st.write("""
    #### Метод Ньютона при вычислении матрицы Якоби для решения системы нелинейных уравнений 
    
    ##### Метод Ньютона для решения системы нелинейных уравнений:

    F(x) = 0,

    где F - вектор-функция: $R^{n}\Rightarrow R^{n}$, $F_{i}$ - компонента вектора-функции -> некая функция на строке i, x - вектор $R^{n}$ неизвестных переменных, основан на следующей итерационной формуле:

    $x^{k+1} = x^{k} - J^{-1}(x^{k}) * F(x^{k})$ ,
    
    где:

    $x^{k}$ - приближение на шаге k,
    
    $x^{k+1}$ - новое приближение,
    
    $J(x^{k})$ - матрица Якоби, вычисленная в точке $x^{k}$,
    
    $J^{-1}(x^{k})$ - обратная матрица к $J(x^{k})$.
    
    ###### Матрица Якоби
    
    Матрица Якоби J(x) -- матрица частных производных компонент функции F по переменным x:
    
    """)

    col21, mid22, col22 = st.columns([1,13,20])
    with col21:
        st.image('gr22.png', width=300)
    with col22:
        st.write('Наглядный пример\n\n источник: https://ru.wikipedia.org/wiki/Матрица_Якоби')

    st.write("""
    Для каждой компоненты $F_{i}$ матрица содержит столбец частных производных по всем переменным $x_{1}, ..., x_{n}$.
    
    ###### Вычисление матрицы Якоби

    Аналитическое дифференцирование

    Численное дифференцирование

    Автоматическое дифференцирование
    
    ##### Особенности применения метода Ньютона
    
    Метод Ньютона сходится локально, поэтому начальное приближение должно быть достаточно близко к решению.

    На каждой итерации необходимо убедиться, что матрица Якоби обратима.

    На каждой итерации надо вычислить матрицу Якоби (либо подставить значения). Особенно затратно для больших систем.
    """)




def display_theory_noscipy():
    st.subheader("Решение задания 4.2")
    st.write("""    
    ##### Исходный код
    """)
    
    code1 = """
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

            print(f"Найденное решение: {solution}")
            print(f"Точное решение: {exact_solution}")
            print(f"Ошибка: {error:.2e}")
            print(f"Количество итераций: {iterations}")

        except ValueError as e:
            print(f"Ошибка: {e}")


if __name__ == "__main__":
    n_values = [3, 5, 10]
    solve_for_different_n(n_values)    
    """

    st.code(code1)
    res = pypy.execute_python_code(code1)
    
    st.write(res)
    
    st.write("""
    ##### Комментарий к коду
    
    Функция создает систему нелинейных уравнений размерности n. Возвращает функцию, которая принимает и возвращает numpy массив.
    
    Внутренняя функция, определяющая систему уравнений. Создает массив нулей размера n.
    
    Определяем первое уравнение системы.
    
    Определяем средние уравнения системы в цикле.
    
    Определяем последнее уравнение системы.

    Функция для численного вычисления матрицы Якоби. 
    
    Инициализация матрицы Якоби и вычисление значения функции в точке x.
    
    Вычисление частных производных численным методом для каждой переменной.

    Функция метода Ньютона для решения системы нелинейных уравнений.
    
    Проверка условия сходимости.
    
    Вычисление матрицы Якоби и решение линейной системы для получения приращения.

    Функция для решения системы с разными размерностями.
    
    Для каждой размерности создаем систему и начальное приближение.
    Находим численное решение, сравниваем с точным решением и вычисляем ошибку.
    
    Выводим результаты.

    Запускаем решение для систем размерности 3, 5 и 10.
    """)

    

def display_theory_scipy():
    st.subheader("Решение задания 4.2 с использованием библиотеки Scipy")
    st.write("""

    ##### Исходный код

    """)
    
    code1 = """
import numpy as np
from scipy.optimize import root

def scipy_solver(n):
    # Система уравнений
    def F(x):
        f = np.zeros(n)
        f[0] = (3 + 2 * x[0]) * x[0] - 2 * x[1] - 3
        for i in range(1, n-1):
            f[i] = (3 + 2 * x[i]) * x[i-1] - 2 * x[i+1] - 2
        f[-1] = (3 + 2 * x[-1]) * x[-2] - 4
        return f

    x0 = np.ones(n)  # Начальное приближение
    result = root(F, x0, method='hybr')  # Можно также использовать 'lm', 'broyden1' и другие методы
    if result.success:
        return result.x
    else:
        raise ValueError("SciPy не нашел решение")

# Проверка на примере n = 5
n = 5
solution_scipy = scipy_solver(n)
print("Решение с SciPy:", solution_scipy)
    
    """
    st.code(code1)
    res = pypy.execute_python_code(code1)
    st.write(res)

def display_task_working():
    #st.subheader("")
    
    tempstr = r"""
    ## Задание 4.2

    ### Введите значения параметров
    
    Приближенное решение системы

    $\begin{aligned}
    (3 + 2x_1) x_1 - 2 x_2 = 3,
    \end{aligned}$

    $\begin{aligned}
    (3 + 2x_i) x_i - x_{i-1} - 2 x_{i+1} = 2,
    \quad i = 2, 3, ..., n-1,
    \end{aligned}$

    $\begin{aligned}
    (3 + 2x_n) x_n - x_{n-1} = 4
    \end{aligned}$
    
    """

    st.write(tempstr)

    
    st.write("""
    #### Ввод параметров решения без scipy
    """)
    
    col5, col6=  st.columns(2)
    n_val = col5.number_input( 'Количество шагов n', value=3, step=1, key="antiduplicat")

    st.write("""
    #### Ввод параметров решения scipy
    """)
    
    col7, col8=  st.columns(2)
    n_val1 = col7.number_input( 'Количество шагов n', value=3, step=1, key="antiduplicat1")

    st.write("""
    #### Результаты работы
    """)

    
    result = countexample_metod1(n_val)
    st.write("""вывод программы без scipy""")
    st.write(result)
    #resres = eval(evaluate_single_fragment(result))
    #st.write(resres )
    
    result1 = countexample_metod2(n_val1)
    st.write("""вывод программы scipy""")
    st.write(result1)
    #resres1 = eval(evaluate_single_fragment(result1))
    #st.write(resres1 )

    #fig, ax = plt.subplots(figsize=(10, 6))
    #diff = np.array(resres) - np.array(resres1)

    #ax.scatter(range(len(resres)), resres, color='blue', label='Метод Ньютона')
    #ax.scatter(range(len(resres1)), resres1, color='red', label='SciPy')
    #ax.scatter(range(len(diff)), diff, color='green', label='Разница')

    #ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    
    #ax.set_title('Сравнение решений')
    #ax.set_xlabel('Индекс переменной')
    #ax.set_ylabel('Значение переменной')
    
    #ax.legend()
    #st.pyplot(fig)



def countexample_metod1(n: int):
    f = io.StringIO()
    with redirect_stdout(f):
        pNNM.callme(n)
    output_string = f.getvalue()
    return output_string

def countexample_metod2(n: int):
    f = io.StringIO()
    with redirect_stdout(f):
        p4_2.scipy_solver(n)
    output_string = f.getvalue()
    return output_string

def display_extra():
    st.subheader("Дополнительная информация о задании 4.2")
    st.write("""
    Программа решатель 
    
    - Скорых Александр
    
    Программа решатель с SciPy
    
    - Тищенко Илья
    
    Презентация в streamlit
    
    - Хаметов Марк
    
    """)