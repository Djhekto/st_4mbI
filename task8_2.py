import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io
from contextlib import redirect_stdout

from scipy.sparse.linalg import cg, LinearOperator

import pypy
import p8_2




def exact_solution(x):
    return 3*x
def run_task8_2():
    
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

**Задание 8.2**

Напишите программу для численного решения
интегрального уравнения Фредгольма первого рода
методом квадратур с использованием квадратурной формулы прямоугольников
при равномерном разбиении интервала интегрирования
с симметризацией матрицы системы уравнений и ее итерационном
решении методом сопряженных градиентов.

С~помощью этой программы найдите приближенное
решение интегрального уравнения

$\\begin{aligned}
  \int_{-1}^{1} |(x-s)| y(s) ds = x^2
\end{aligned}$

при различной точности приближенного решения системы линейных
уравнений и сравните его с точным решением интегрального уравнения.

Решите также эту задачу с помощью библиотеки SciPy.
    """)

def display_theory():
    st.subheader("О методе прямоугольников")
    st.write("""
    ##### Разделение интервала
    
    Разбиение интервала интегрирования $ [a, b] $ на $ n $ равных отрезков
    """)
    
    st.latex("""
    \Delta x = \\frac{b - a}{n}
    """)
    
    
    st.write("""

    ##### Применение квадратурной формулы прямоугольников: 

    Квадратурная формула прямоугольников предполагает использование значений функции на левых или правых концах отрезков для оценки интеграла.

    """)
    
    st.latex("""
    I \\approx \sum_{i=0}^{n-1} f(x_i) \Delta x
    """)

    st.write("""

   где $x_i = a + i \Delta x$ (например, для левых прямоугольников).

- **Симметризация матрицы:**

    Есть система уравнений, описываемая матрицей $A$, важно, чтобы эта матрица была симметричной. 

    Это может быть достигнуто, например, путем усреднения:

    """)
    
    st.latex("""
    A_{sym} = \\frac{1}{2}(A + A^T)
    """)

    st.write("""

   где $A^T$ — транспонированная матрица.

- **Итерационное решение методом сопряженных градиентов:** После симметризации матрицы можно использовать метод сопряженных градиентов для решения системы уравнений \(A_{sym}x = b\). Алгоритм работы заключается в итеративном улучшении приближения к решению \(x\) путем минимизации функции:

https://ru.wikipedia.org/wiki/Метод_сопряжённых_градиентов_(СЛАУ)

    """)
    
    st.latex("""
    \Phi(x) = \\frac{1}{2}x^T A_{sym} x - b^T x
    """)

    st.write("""

   Итерации продолжаются до достижения заданной точности.

- **Сходимость:** Сходимость данного метода зависит от свойств матрицы $A_{sym}$, таких как положительная определенность и условность. 

    Если матрица хорошо обусловлена, итерационный метод будет сойтись быстрее.

             """)


def display_theory111111111111():
    st.subheader("О методе прямоугольников")
    st.write(f"""
             Вот теоретический обзор квадратурной формулы прямоугольников при равномерном разбиении интервала интегрирования с симметризацией матрицы системы уравнений и ее итерационном решении методом сопряженных градиентов в формате Markdown:

### Квадратурная формула прямоугольников

Квадратурная формула прямоугольников - это числовой метод для приближения определенного интеграла. Он основан на разбиении интервала интегрирования на равномерные прямоугольники и использовании среднего значения функции в каждом прямоугольнике для оценки интеграла.

### Равномерное разбиение интервала

При равномерном разбиении интервала [a, b] на n равных прямоугольников, ширина каждого прямоугольника равна:
    """)
    
    st.latex("""
            h = (b - a) / n
             """)    

    st.write(f"""
### Симметризация матрицы системы уравнений

Симметризация матрицы системы уравнений - это процесс, который делает матрицу симметричной, сохраняя при этом ее первоначальную структуру и свойства. Это важно для некоторых числовых методов, особенно в контексте метода сопряженных градиентов.

### Метод сопряженных градиентов

Метод сопряженных градиентов (CG) - это эффективный итерационный метод для решения систем линейных уравнений. Он основан на поиске направления спуска, которое минимизирует неопределенность системы.

### Применение к квадратурной формуле прямоугольников

При применении метода сопряженных градиентов к квадратурной формуле прямоугольников с симметризацией матрицы, мы можем достичь следующего:

1. Разбием интервал [a, b] на n равномерных прямоугольников.
2. Для каждой точки разбиения вычисляем среднее значение функции.
3. Симметризуем матрицу системы уравнений, связанной с интегральным отношением.
4. Применяем метод сопряженных градиентов для решения системы уравнений.

### Алгоритм

1. Разбиваем интервал [a, b] на n равномерные прямоугольники:
   $x_i = a + i * h$, где $i = 0, 1, ..., n$

2. Вычисляем среднее значение функции f(x_i):

   $f_i = f(x_i)$

3. Создаем матрицу A и вектор b для системы уравнений:

   $A = [f_0, ..., f_n]$
   $b = [h * f_0, ..., h * f_n]$

4. Симметризуем матрицу A:

   $A' = 0.5 * (A + A^T)$

5. Применяем метод сопряженных градиентов:

   $x_k+1 = x_k - (A'^T * A')^-1 * A'^T * b$

6. Повторяем шаг 5 до достижения требуемой точности.

### Преимущества и недостатки

Преимущества:
- Простота реализации
- Эффективность при правильном выборе n
- Возможность использования симметризации для улучшения стабильности метода

Недостатки:
- Зависимость от качества разбиения интервала
- Потенциальная потеря точности при большом n

### Заключение

Квадратурная формула прямоугольников с использованием симметризованной матрицы и метода сопряженных градиентов представляет собой эффективный подход к численному интегрированию. Этот метод сочетает в себе простоту реализации с высокой точностью, что делает его полезным инструментом в различных областях вычислительной математики.

    """)

def display_theory_noscipy():
    st.subheader("Решение задания 8.2")
    st.write("""
    ##### Пример кода
    """)
    
    st.code("""
import numpy as np


def kernel(x, s):
    #Ядро интегрального уравнения: |x-s|
    return abs(x - s)


def right_side(x):
    #Правая часть уравнения: x^2
    return x ** 2


def exact_solution(x):
    #Точное решение уравнения: y(x) = 3x
    return 3 * x


def conjugate_gradient_method(A, b, tol):
    
    #Метод сопряженных градиентов для решения системы Ax = b
    #A - симметричная матрица
    #b - вектор правой части
    #tol - требуемая точность
    
    x = np.zeros_like(b)
    r = b - np.dot(A, x)
    p = r.copy()
    r_norm_sq = np.dot(r, r)

    iteration = 0
    max_iter = len(b) * 2  # максимальное число итераций

    while iteration < max_iter:
        Ap = np.dot(A, p)
        alpha = r_norm_sq / np.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        r_norm_sq_new = np.dot(r, r)

        # Проверка сходимости
        if np.sqrt(r_norm_sq_new) < tol:
            break

        beta = r_norm_sq_new / r_norm_sq
        p = r + beta * p
        r_norm_sq = r_norm_sq_new
        iteration += 1

    return x, iteration


def solve_fredholm(n, tol):
    
    #Решение интегрального уравнения методом квадратур
    #n - количество точек разбиения
    #tol - требуемая точность решения
    
    # Шаг интегрирования
    h = 2 / n

    # Узлы квадратурной формулы
    x = np.linspace(-1, 1, n)

    # Формируем матрицу системы
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            A[i, j] = kernel(x[i], x[j]) * h

    # Формируем вектор правой части
    b = right_side(x)

    # Симметризуем систему
    A_sym = np.dot(A.T, A)
    b_sym = np.dot(A.T, b)

    # Решаем систему методом сопряженных градиентов
    y, iterations = conjugate_gradient_method(A_sym, b_sym, tol)

    return x, y, iterations


def calculate_error(x, y_approx, y_exact):
    #Вычисление максимальной абсолютной погрешности
    return np.max(np.abs(y_approx - y_exact))


def main():
    try:
        # Ввод параметров
        n = int(input("Введите количество точек разбиения (n > 1): "))


        tol = float(input("Введите требуемую точность (например, 1e-6): "))


        # Решение уравнения
        x, y_approx, iterations = solve_fredholm(n, tol)

        # Вычисление точного решения
        y_exact = exact_solution(x)

        # Вычисление погрешности
        error = calculate_error(x, y_approx, y_exact)

        # Вывод результатов
        print("\\nРезультаты:")
        print(f"Количество итераций: {iterations}")
        print(f"Максимальная абсолютная погрешность: {error:.2e}")
        print("\\nСравнение решений:")
        print("    x      Приближенное  Точное")
        print("-" * 40)
        for i in range(n):
            print(f"{x[i]:7.3f} {y_approx[i]:12.6f} {y_exact[i]:12.6f}")


    except ValueError:
        print("Ошибка ввода! Проверьте формат данных.")
    except np.linalg.LinAlgError:
        print("Ошибка! Не удалось решить систему уравнений.")
    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")


if __name__ == "__main__":
    main()
            """)

def display_theory_scipy():
    st.subheader("Решение задания 8.2 с использованием библиотеки Scipy")
    st.write("""
    """)
    st.subheader("Решение задания 8.2")
    st.write("""
    ##### Пример кода
    """)
    
    st.code("""
import numpy as np
from scipy.sparse.linalg import cg, LinearOperator
import matplotlib.pyplot as plt

def exact_solution(x):
    return 3*x

# Параметры разбиения
N = 100
a, b = -1, 1
h = (b - a)/N
s = np.linspace(a, b, N+1)

# Правая часть: f(x) = x²
f = s**2

# Формируем матрицу A: A_{ij} = |s_i - s_j|
S_i, S_j = np.meshgrid(s, s, indexing='ij')
A = np.abs(S_i - S_j)

# Система: A y = f/h
rhs = f/h

# Оператор для метода сопряжённых градиентов
def mv(x):
    return A.dot(x)
A_op = LinearOperator((N+1, N+1), matvec=mv)

tol = 1e-10
y_approx, info = cg(A_op, rhs, atol=tol)

if info == 0:
    print("Система решена методом сопряжённых градиентов с заданной точностью.")
else:
    print("Метод сопряжённых градиентов не достиг требуемой точности.")

# Невязка
res = A.dot(y_approx) - rhs
err = np.linalg.norm(res)
print(f"Норма невязки: {err}")

# Сравниваем с «точным» решением y(x)=3x
y_exact = exact_solution(s)
error_norm = np.linalg.norm(y_approx - y_exact)
print(f"Норма ошибки между численным решением и точным y=3x: {error_norm}")

# Построение графиков
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(s, y_approx, label='Численное решение y(s)')
plt.plot(s, y_exact, label='Точное решение y=3x', linestyle='--')
plt.xlabel('s')
plt.ylabel('y(s)')
plt.grid(True)
plt.legend()
plt.title("Сравнение решений")

# Проверим, что интеграл от найденного решения приближает x²:
I_approx = A.dot(y_approx)*h

plt.subplot(1,2,2)
plt.plot(s, s**2, label='x² (правая часть)')
plt.plot(s, I_approx, label='Интеграл с приближённым y(s)')
plt.xlabel('x')
plt.ylabel('значение')
plt.grid(True)
plt.legend()
plt.title("Проверка интегрального уравнения")

plt.tight_layout()
plt.show()

            """)
    
def display_task_working():
    st.title("Численное решение интегрального уравнения Фредгольма 1-го рода")
    st.subheader("C использованием библиотеки Scipy:")
    st.write(r"Решаем уравнение: $\int_{-1}^{1}|x-s|y(s)ds = x^2$, предполагая точное решение $y(x)=3x$")

    N = st.sidebar.slider("Число разбиений (N):", min_value=50, max_value=500, value=100)
    a, b = -1, 1
    h = (b - a)/N
    s = np.linspace(a, b, N+1)

    # Правая часть: f(x) = x²
    f = s**2

    # Формируем матрицу A: A_{ij} = |s_i - s_j|
    S_i, S_j = np.meshgrid(s, s, indexing='ij')
    A = np.abs(S_i - S_j).astype(float)  # Приведём к float на всякий случай

    # Система: A y = f/h
    rhs = (f/h).astype(float)

    # Создаём линейный оператор для метода cg
    def mv(x):
        return A.dot(x)
    A_op = LinearOperator((N+1, N+1), matvec=mv)

    tol = 1e-10
    try:
        y_approx, info = cg(A_op, rhs, atol=tol)
        if info == 0:
            st.write("Система решена методом сопряжённых градиентов с заданной точностью.")
        else:
            st.write(f"Метод сопряжённых градиентов вернул info={info}, возможно не достигнута требуемая точность.")
    except Exception as e:
        st.error(f"Ошибка при решении СЛАУ методом CG: {e}")
        return

    # Проверим невязку
    res = A.dot(y_approx) - rhs
    err = np.linalg.norm(res)
    st.write(f"Норма невязки: {err}")

    # Сравниваем с «точным» решением y(x)=3x
    y_exact = exact_solution(s)
    error_norm = np.linalg.norm(y_approx - y_exact)
    st.write(f"Норма ошибки между численным решением и точным y=3x: {error_norm}")

    #=====
    printme1, res1_yaprox, res_x = countexample_metod1(30)




    fig, ax = plt.subplots()
    ax.plot(s, y_approx, label='$Численное решение y(s)$')
    ax.plot(s, y_exact, label='$Точное\;решение\;y=3x$', linestyle='--')
    ax.plot(res_x, res1_yaprox, label='$ решение без scipy $', linestyle='--')
    ax.set_xlabel('s')
    ax.set_ylabel('y(s)')
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    # Проверим, что интеграл от найденного решения приближает x²:
    I_approx = A.dot(y_approx)*h
    fig2, ax2 = plt.subplots()
    ax2.plot(s, s**2, label='$x^2\;(правая\;часть)$',marker='x')
    ax2.plot(s, I_approx, label='$Интеграл\;с\;приближённым\;y(s)$')
    ax2.set_xlabel('$x$')
    ax2.set_ylabel('$значение$')
    ax2.grid(True)
    ax2.legend()
    st.pyplot(fig2)

    st.write("# решение без scipy")

    st.write(printme1)
        






    if __name__ == "__main__":
        display_task_working()

    


def countexample_metod1(n: int):
    f = io.StringIO()
    with redirect_stdout(f):
        res, resx = p8_2.main11(n)
    output_string = f.getvalue()
    return output_string, res, resx


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
