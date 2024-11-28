import streamlit as st
import numpy as np
import pNNM

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
    st.subheader("Решение задания 4.1")
    st.write("""
    4.2
    
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
    st.subheader("Решение задания 4.1 с использованием библиотеки Scipy")
    st.write("""


    """)

def display_task_working():
    st.subheader("Т")
    st.write("""
    #### Ввод параметров решения без scipy
    """)
    
    col1, col2 = st.columns(2)
    interval_start = col1.number_input('Начало интервала: a', value=0.0, step=0.001, format="%0.5f", key="antidupli5")
    interval_end = col2.number_input('Конец интервала: b', value=10.0, step=0.001, format="%0.5f", key="antidupli6")
    epsilon = col1.number_input('Точность: $\\varepsilon$', value=1e-6, step=0.001, format="%0.7f", key="antidupli7")
    max_iter = col2.number_input('Максимальное количество итераций', value=1000, step=1, key="antidupli8")

    st.write("""
    #### Ввод параметров решения scipy
    """)
    
    #col3, col4 = st.columns(2)
    #interval_start1 = col3.number_input( 'Начало интервала: a', value=0.0, step=0.001, format="%0.5f", key="antidupli1")
    #interval_end1 = col4.number_input( 'Конец интервала: b', value=10.0, step=0.001, format="%0.5f", key="antidupli2")
    #epsilon1 = col3.number_input( 'Точность: $\\varepsilon$', value=1e-6, step=0.001, format="%0.7f", key="antidupli3")
    #max_iter1 = col4.number_input( 'Максимальное количество итераций', value=1000, step=1, key="antidupli4")

    col5, col6 = st.columns(2)
    sc_dimention_c = col5.number_input( 'Размернось системы', value=5, step=1, key="antidupli4")
    countexample_metod2(sc_dimention_c)


def countexample_metod2 ( n: int ):
    pNNM.callme(n)
    
    return



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