import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


import pypy
import p8_1
import ptask8_1skipy


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
    ptask8_1skipy.run_task8_1()
#    st.subheader("Решение задания 4.1 с использованием библиотеки Scipy")
#    st.write("""
#    """)
    
def display_task_working():
    st.write("""
             ФФАААААААААААААААААААААААААААААААААААААА
             А Я ХЗ КАК ОРГАНИЗОВОВАТЬ, ЕСЛИ У ОДНОГО ПРОГА У ДРУГОГО ПРИЛОЖЕНИЕ СТРИМЛИТ ААААААААААААААААААААА
    """)    


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
