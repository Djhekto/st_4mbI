import streamlit as st
import numpy as np

#import math
import matplotlib.pyplot as plt

#from PIL import Image
#import base64

import pNM
import p4_1


def run_task4_1():
    
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
    st.subheader("Задание 4.1")
    st.write("""
    ##### 4 Нелинейные уравнения и системы

    **Задание 4.1**

    Напишите программу для нахождения решения нелинейного уравнения
    $f(x) = 0$ методом бисекций.

    С ее помощью найдите корни уравнения

    $ \\begin{aligned}
    (1+x^2) e^{-x} + \sin(x) = 0
    \end{aligned}
    $

    на интервале [0,10].

    Решите также эту задачу с помощью библиотеки SciPy.
    """)

def display_theory():
    st.subheader("О методе бисекций")
    st.write(f"""
    Метод бисекций ( метод деления пополам) - это численный метод для поиска корня функции в заданном интервале.

    ##### 1. Принцип работы:
    - Метод работает на замкнутом интервале [a, b], где функция имеет противоположные знаки в точках a и b.
    - На каждой итерации выбирается середина интервала c = (a + b)/2.
    - Если функция меняет знак в точке c, то новый интервал становится [c, b] или [a, c] в зависимости от знака функции в точке c.
        - то есть: находим знак f(c). Выбираем половинный интервал в сторону границы с противоположным знаком, эта граница не изменяется при изменении интервала
        """)
    
    col21, mid22, col22 = st.columns([1,10,20])
    with col21:
        st.image('gr12.png', width=300)
    with col22:
        st.write('Наглядный пример\n\n источник: https://ru.wikipedia.org/wiki/Метод_бисекции#/media/Файл:Bisection_method.svg')
    
    st.write("""
    - Процесс повторяется до достижения желаемой точности.

    ##### 2. Преимущества:
    - Гарантированная сходимость при условии существования корня в интервале.
    - Простота реализации.
    - Не требует вычисления производной функции.

    ##### 3. Недостатки:
    - Медленная сходимость (порядок O(log(n)), где n - число итераций).
    - Требует начального интервала, содержащего корень.

    ##### 4. Применение:
    - Эффективен для простых функций с одним корнем в интервале.
    - Часто используется как вспомогательный метод в более сложных алгоритмах.

    ##### 5. Реализация:
    - Обычно реализуется через рекурсивную функцию или цикл.
    - Критерий остановки: достижение заданной точности или максимального числа итераций.

    Метод бисекций является базовым численным методом для решения нелинейных уравнений и часто используется в качестве первого приближения или для проверки других методов.
    """)

def display_theory_noscipy():
    st.subheader("Решение задания 4.1")
    st.write("""
    f(x) = (1 + x^2) * e^(-x) + sin(x)
    """)

def display_theory_scipy():
    st.subheader("Решение задания 4.1 с использованием библиотеки Scipy")
    st.write("""
    f(x) = (1 + x^2) * e^(-x) + sin(x)
    """)

def display_task_working():
    st.write("""
    ## Задание 4.1

    ### Введите значения параметров
    
    Для уравнения
    $ \\begin{aligned}
    (1+x^2) e^{-x} + \sin(x) = 0
    \end{aligned}
    $
    на интервале [a,b].

    
    """)

    #st.header("Задание 4.1")
    
    #st.subheader("Введите значения параметров ")
    
    st.write("""
    #### Ввод параметров решения без scipy
    """)
    
    col1, col2 = st.columns(2)
    interval_start = col1.number_input('Начало интервала: a', value=0.0, step=0.001, format="%0.5f")
    interval_end = col2.number_input('Конец интервала: b', value=10.0, step=0.001, format="%0.5f")
    epsilon = col1.number_input('Точность: $\\varepsilon$', value=1e-6, step=0.001, format="%0.7f")
    max_iter = col2.number_input('Максимальное количество итераций', value=1000, step=1)

    st.write("""
    #### Ввод параметров решения scipy
    """)
    
    col3, col4 = st.columns(2)
    interval_start1 = col3.number_input( 'Начало интервала: a', value=0.0, step=0.001, format="%0.5f", key="antiduplicate1")
    interval_end1 = col4.number_input( 'Конец интервала: b', value=10.0, step=0.001, format="%0.5f", key="antiduplicate2")
    epsilon1 = col3.number_input( 'Точность: $\\varepsilon$', value=1e-6, step=0.001, format="%0.7f", key="antiduplicate3")
    max_iter1 = col4.number_input( 'Максимальное количество итераций', value=1000, step=1, key="antiduplicate4")

    printme, res1 = countexample_metod1(interval_start,interval_end,epsilon,max_iter)
    strtemp, res2 = countexample_metod2(interval_start1,interval_end1,epsilon1,max_iter1)
    printme = printme + strtemp + str(res2) +"\n\n"
    res3 = [abs(res1[i] - res2[i]) for i in range(len(res1))]
    
    printme = printme + "\n##### Разница значений методов: \n\n" + str(res3) +"\n\n"
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(range(len(res1)), res1, color='blue', label='Метод бисекций')
    ax.scatter(range(len(res2)), res2, color='red', label='SciPy метод')
    ax.scatter(range(len(res3)), res3, color='purple', label='Разница результатов')
    ax.set_xlabel('Номер корня')
    ax.set_ylabel('Значение корня')
    ax.set_title('Сравнение корней уравнения')
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)
    
    st.write(printme)
    #st.write(abs(res1 - res2))
    

def countexample_metod1 (  a: float, b: float , eps:  float, maxiter: int ):
    #equation = lambda x: (1 + x ** 2) * math.exp(-x) + math.sin(x)

    intervals = pNM.find_intervals(pNM.equation, a, b, maxiter)
    
    strtoprintres = "##### =Решение без SciPy= \n\n" + "Найденные корни уравнения "
    strtoprintres = strtoprintres + " $(1+x^{2})e^{-x} + sin(x)=0$ :\n\n"
    #strtoprintres = strtoprintres + "\nx         f(x)              Итераций\n\n"
    #strtoprintres = strtoprintres + "-" * 45 + "\n\n"
    
    lroots, literations = [], []
    for aa, bb in intervals:
        try:
            root, iterations = pNM.bisection_method(pNM.equation, aa, bb,eps, maxiter)
            lroots.append(root)
            literations.append(iterations)
        except ValueError as e:
            print(f"Ошибка на интервале [{a}, {b}]: {e}\n\n")

    #print(f"{root:.6f}  {pNM.equation(root):15.2e}  {iterations:8d}")
    
    #strtoprintres = strtoprintres + f"{root:.6f}  {pNM.equation(pNM.root):15.2e}  {iterations:8d} \n"

    strtoprintres = strtoprintres + str(lroots) + " за такое количество итераций на разбитом интервале: " + str(literations) + "   \n"

    return strtoprintres, lroots

def countexample_metod2 (  a: float, b: float , eps:  float, maxiter: int ):
    res = p4_1.mainfunc(a,b,eps,maxiter)
    return "\n ##### =Решение через SciPy= \n\n"+"Найдены корни уравнения:\n\n", res


def display_extra():
    st.subheader("Дополнительная информация о задании 4.1")
    st.write("""
    Программа решатель 
    
    - Скорых Александр
    
    Программа решатель с SciPy
    
    - Тищенко Илья
    
    Презентация в streamlit
    
    - Хаметов Марк
    
    """)
    #st.image("logo.png")
