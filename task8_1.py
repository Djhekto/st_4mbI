import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io
from contextlib import redirect_stdout


from scipy.linalg import solve


import pypy
import p8_1
#import ptask8_1skipy


def run_task8_1():
    
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
    st.subheader("Задание 8.1")
    st.write("""
    ##### 8 Интегральные уравнения

**Задание 8.1**

Напишите программу для численного решения
интегрального уравнения Фредгольма второго рода
методом квадратур с использованием квадратурной формулы трапеций
при равномерном разбиении интервала интегрирования.

С помощью этой программы найдите приближенное
решение интегрального уравнения

$ \\begin{aligned}
  y(x) - \int_{-1}^{1} \\frac{1}{\pi (1+(x-s)^2)} y(s) ds = 1
\end{aligned}
$

при различном числе частичных отрезков.

Решите также эту задачу с помощью библиотеки SciPy.
    """)

def display_theory():
    st.subheader("О методе трапеций")
    st.write(f"""
    Квадратурная формула трапеций - это численный метод для приближенного вычисления определенных интегралов. При равномерном разбиении интервала интегрирования эта формула имеет следующий вид:

    """)
    
    col21, mid22, col22 = st.columns([1,10,20])
    with col21:
        st.image('81t.png', width=200)
    with col22:
        st.write('Наглядный пример\n\n источник: https://ru.wikipedia.org/wiki/Метод_трапеций')
    
    st.write(f"""
    ### Общая форма формулы

    Для функции f(x) на интервале [a, b] с n равными подинтервалами (n > 0):
    """)
    
    #\int_a^b f(x) \, dx \\approx \\frac{b - a}{2} (f(a) + f(b))
    #             I \\approx \\frac{h}{2} \left[ f(a) + f(b) + 2\sum_{i=1}^{n-1} f(x_i) \\right],
    
    st.latex("""
             \int_a^b f(x) \, dx \\approx \\frac{b - a}{2} (f(a) + f(b))
             """)
    
    #st.latex("""
    #        где\\ x_i = a + i\cdot h \\text{ и } i = 1, 2, \ldots, n-1,
    #         """)
    
    #st.latex("""
    #        h = \\frac{b-a}{n} - длина каждого подинтервала.
    #         """)
    
    st.write(f"""
    ### Детальное объяснение

    - Интервал [a, b] делится на n равных частей шириной h = (b-a)/n.

    - В каждом подинтервале функция аппроксимируется отрезком прямой, соединяющим значения функции в конечных точках подинтервала.

    - Площадь трапеции, образованной этим отрезком и осью x, используется как приближение площади под кривой в этом подинтервале.

    - Сумма площадей всех таких трапеций дает приближенное значение интеграла.

    ### Погрешность
    
    Основная формула для ошибочного члена может быть выражена следующим образом:
    
    """)
    
    st.latex("""
    E_T = -\\frac{(b - a)^3}{12} f''(\\xi)
             """)

    st.write(f"""
    где 
    $E_T$— это погрешность, 
    $\\xi$— некоторая точка в интервале 
    , а 
    $f''(\\xi)$ — значение второй производной функции в этой точке.
    
    #### Вторая производная: 
    Для точной оценки погрешности необходимо знать или оценить величину нахождения второй производной
    
    погрешность:
    """)

    st.latex("""
    |E_T| \leq \\frac{(b - a)^3}{12n^2} \max |f''(x)|
             """)
    
    st.write(f"""
             ###### Что-то непонятное в википедии
             
             https://ru.wikipedia.org/wiki/Метод_трапеций
    """)
    
    st.image("811t.png")
    
    st.write(f"""
    
    #### Общие слова

    Погрешность формулы трапеций зависит от числа подинтервалов n и от второй производной интегрируемой функции. Чем больше n, тем меньше погрешность. Для гладких функций погрешность уменьшается пропорционально 1/n^2.

    Формула трапеций проста в реализации и эффективна для многих практических задач, особенно когда требуется быстрое приближение или когда точный интеграл трудно вычислить аналитически.

    """)

def display_theory_noscipy():
    st.subheader("Решение задания 8.1")
    st.write("""
    ##### Пример реализации
    """)
    
    st.code("""
import numpy as np
from math import pi


def kernel(x, s):
    #Ядро интегрального уравнения
    return 1 / (pi * (1 + (x - s) ** 2))


def free_term(x):
    #Свободный член уравнения
    return 1


def solve_fredholm(n):
    #Решение интегрального уравнения методом квадратур
    #n - количество отрезков разбиения
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
            """)
    

def display_theory_scipy():
    st.subheader("Решение задания 8.1 c использованием scipy")
    st.write("""
    ##### Пример реализации
    """)
    
    st.code("""
import numpy as np
from scipy.linalg import solve
import matplotlib.pyplot as plt

# Ядро
def K(x, s):
    return 1.0/(np.pi*(1+(x-s)**2))

def build_weights(N, a, b, method):
    h = (b - a)/N
    if method == "Trapezoidal":
        # Формула трапеций
        w = np.full(N+1, h)
        w[0] = h/2
        w[-1] = h/2
    elif method == "Rectangle (left)":
        # Левые прямоугольники
        w = np.zeros(N+1)
        w[:N] = h  # последний узел не нужен, так как интеграл покрыт N прямоугольниками
    elif method == "Simpson":
        # Формула Симпсона для чётного N
        if N % 2 != 0:
            raise ValueError("Для метода Симпсона N должно быть чётным.")
        w = np.zeros(N+1)
        w[0] = h/3
        w[-1] = h/3
        for j in range(1, N):
            if j % 2 == 1:
                w[j] = 4*h/3
            else:
                w[j] = 2*h/3
    else:
        raise ValueError("Неизвестный метод квадратур.")
    return w

def solve_fredholm_second_kind(N, method):
    a, b = -1, 1
    x_nodes = np.linspace(a, b, N+1)
    w = build_weights(N, a, b, method)
    
    # Матрица A = I - K_w
    A = np.zeros((N+1, N+1))
    for i in range(N+1):
        for j in range(N+1):
            A[i,j] = -K(x_nodes[i], x_nodes[j])*w[j]
        A[i,i] += 1.0
    
    f = np.ones(N+1)
    y = solve(A, f)
    return x_nodes, y, w

def compute_residual(x_nodes, y, w):
    # Вычислим невязку: r(x_i) = y(x_i) - 1 - ∫K(x_i,s)*y(s)ds
    # ≈ y(x_i) - 1 - Σ_j K(x_i,x_j)*y(x_j)*w_j
    N = len(x_nodes)-1
    res = np.zeros(N+1)
    for i in range(N+1):
        I_approx = np.sum(K(x_nodes[i], x_nodes)*y*w)
        res[i] = y[i] - I_approx - 1.0
    return res

if __name__ == "__main__":
    # Пример использования:
    N_values = [50, 100]
    methods = ["Trapezoidal", "Rectangle (left)", "Simpson"]

    plt.figure(figsize=(10,8))

    for i, method in enumerate(methods):
        plt.subplot(len(methods), 1, i+1)
        for N in N_values:
            # Для Симпсона убедимся, что N чётное
            if method == "Simpson" and N%2!=0:
                continue
            
            x_nodes, y, w = solve_fredholm_second_kind(N, method)
            res = compute_residual(x_nodes, y, w)
            err_norm = np.linalg.norm(res, np.inf)
            
            label = f"N={N}, err={err_norm:.2e}"
            plt.plot(x_nodes, y, label=label)
        
        plt.xlabel('x')
        plt.ylabel('y(x)')
        plt.title(f"Метод: {method}")
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.show()

            """)
    
def display_task_working():
        st.title("Интерактивное решение интегрального уравнения Фредгольма второго рода")
        st.write(r"""
        Рассматриваем уравнение:
        $$
        y(x) - \int_{-1}^{1}\frac{1}{\pi(1+(x-s)^2)}y(s)ds = 1.
        $$""")
        st.subheader("Решение задания c использованием scipy")
        st.write(r"""Выберите параметры разбиения и квадратурную формулу.""")

        
        method = st.sidebar.selectbox("Выберите квадратурную формулу", 
                                      ["Трапеций", "Прямоугольников_(левых)", "Симпсона"])
        if method == "Симпсона":
            N = st.number_input("Количество узлов (n):", min_value=4, max_value=150, value=4, step=2)
        else:
            N = st.number_input("Количество узлов (n):", min_value=3, max_value=150, value=3, step=1)


        N11111 = st.number_input("Для без scipy:", min_value=3, max_value=150, value=3, step=1)

        a, b = -1, 1
        h = (b - a)/N
        
        # Проверка на метод Симпсона
        if method == "Симпсона" and N%2 != 0:
            st.error("Для метода Симпсона число разбиений N должно быть чётным.")
            return

        # Узлы
        x_nodes = np.linspace(a, b, N+1)

        def K(x, s):
            return 1.0/(np.pi*(1+(x-s)**2))

        # Формируем веса в зависимости от метода
        if method == "Трапеций":
            # Трапеций
            w = np.full(N+1, h)
            w[0] = h/2
            w[-1] = h/2
        elif method == "Прямоугольников_(левых)":
            # Левые прямоугольники
            w = np.full(N+1, 0.0)
            w[:N] = h  # последний узел не используется для прямоугольников
        else:
            # Simpson (N чётно)
            w = np.zeros(N+1)
            w[0] = h/3
            w[-1] = h/3
            for j in range(1, N):
                if j%2 == 1:
                    w[j] = 4*h/3
                else:
                    w[j] = 2*h/3

        # Формируем матрицу A = I - K_w
        A = np.zeros((N+1, N+1))
        for i in range(N+1):
            for j in range(N+1):
                A[i,j] = -K(x_nodes[i], x_nodes[j])*w[j]
            A[i,i] += 1.0

        f = np.ones(N+1)
        # Решаем СЛАУ
        y = solve(A, f)

        #============
        printme1, res1_y = countexample_metod1(N11111)
        
        res1_y = [-1*e for e in res1_y]
        #============
        

        # Отобразим результаты
        fig, ax = plt.subplots()
        ax.plot(x_nodes, y, label='$y(x)$')
        #ax.plot(x_nodes, res1_y, label='y(x) no scipy')
        ax.plot(x_nodes, res1_y, label='y(x) no scipy', linestyle='--', alpha=0.5)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y(x)$')
        ax.grid(True)
        ax.set_title(f"$Решение\;уравнения,\;N={N},\;метод={method}$")
        ax.legend()

        st.pyplot(fig)

        # Вычислим невязку
        I_approx = np.zeros(N+1)
        for i in range(N+1):
            I_approx[i] = np.sum(K(x_nodes[i], x_nodes)*y*w)
        residual = y - I_approx - 1.0
        err_norm = np.linalg.norm(residual, np.inf)

        st.write(f"Максимальная абсолютная невязка: {err_norm}")
        if err_norm > 1e-2:
            st.warning("Невязка достаточно велика. Попробуйте увеличить N или сменить метод.")
        
        st.write("##### Решение без scipy\n\n")
        st.write(printme1)
        

def countexample_metod1(n: int):
    f = io.StringIO()
    with redirect_stdout(f):
        res = p8_1.main11(n)
    output_string = f.getvalue()
    return output_string, res
    

    

    


def display_extra():
    st.subheader("Дополнительная информация о задании 4.1")
    st.write("""

    ТУТ НУЖНО ПОМЕНЯТЬ
    
    Программа решатель 
    
    - Скорых Александр
    
    Программа решатель с SciPy
    
    - Тищенко Илья
    
    Презентация в streamlit
    
    - Хаметов Марк
    
    """)
    #st.image("logo.png")
