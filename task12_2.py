import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from matplotlib import cm

import io
from contextlib import redirect_stdout

import pypy
import p12_2


def run_task12_2():
    
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
    st.subheader("Задание 12.2")
    st.write("""
##### 12 Нестационарные задачи математической физики

**Задание 12.2**

Напишите программу численного решения на равномерной
по пространству и времени сетке двумерной задачи

$\\begin{aligned}
  \\frac{\partial u}{\partial t}=
  \\frac{\partial^2 u}{\partial x_1^2} +
  \\frac{\partial^2 u}{\partial x_2^2} +
  f(\\bm x,t) ,
  \quad \\bm x \in \Omega ,
  \quad 0 < t \le T ,
\end{aligned}$

$\\begin{aligned}
  u(\\bm x,t) = 0,
  \quad \\bm x \in \partial \Omega
  \quad 0 < t \le T ,
\end{aligned}$

$\\begin{aligned}
 u(\\bm x,0)=v(\\bm x),
  \quad \\bm x \in \Omega
\end{aligned}$

в прямоугольнике $\Omega$
при использовании схемы переменных направлений
Письмена-Рэкфорда.
Работоспособность  программы продемонстрируйте на численном решении
задачи с точным решением

$\\begin{aligned}
  u(\\bm x,t) = t x_1 (l_1-x_1) x_2 (l_2 - x_2)
\end{aligned}$

на различных сетках по времени.

Решите также эту задачу с помощью библиотеки SciPy.
    """)

def display_theory():
    st.subheader("О системе")
    st.write(f"""
    ##### f
    """)

    st.latex("""
\[

\frac{\partial u}{\partial t} = \frac{\partial^2 u}{\partial x_1^2} + \frac{\partial^2 u}{\partial x_2^2} + f(\bm{x}, t),

\quad \bm{x} \in \Omega, \quad 0 \]

где \( u(\bm{x}, t) \) — это искомая функция (например, температура) в точке \( \bm{x} = (x_1, x_2) \) и времени \( t \), \( f(\bm{x}, t) \) — внешняя сила или источник, а \( \Omega \) — область, в которой мы рассматриваем задачу.
    """)

    st.write("""

### Шаги для анализа и решения задачи:

- **Физический смысл**:

Это уравнение описывает изменение функции \( u \) во времени под воздействием диффузии в двух пространственных направлениях (через вторые производные) и внешнего источника \( f(\\bm${x}$, t) \). Оно может моделировать, например, распространение тепла в двумерной плоскости.

- **Граничные условия**:

Необходимо задать граничные условия на границах области \( \Omega \). Это могут быть:

- Дирихлеевы условия: \( u(\bm{x}, t) = g(\bm{x}, t) \) на границе, где \( g \) — заданная функция;

- Невязочные условия:

    """)

    st.latex("""
\( \frac{\partial u}{\partial n} = h(\bm{x}, t) \) на границе, где \( n \) — нормаль к границе;
    """)

    st.write("""
             

- Начальное условие: \( u(\bm{x}, 0) = u_0(\bm{x}) \) — начальное распределение функции.

- **Методы решения**:

Для решения этого уравнения можно использовать различные методы:

- Метод разделения переменных, если возможно, предполагая, что \( u(\bm{x}, t) = X(x_1)Y(x_2)T(t) \);

- Методы интегрирования по частям и преобразования Фурье для получения решения в частных случаях;

- Численные методы, такие как метод конечных разностей или метод конечных элементов, для решения более сложных задач.

- **Анализ решений**:

После нахождения решения важно проанализировать его поведение во времени и пространстве. Это может включать в себя визуализацию результатов, исследование устойчивости решений и влияние начальных и граничных условий на динамику системы.

Таким образом, данное уравнение является важным инструментом для моделирования процессов, происходящих в двумерных системах, и находит применение в различных областях, таких как теплообмен, гидродинамика и другие физические явления.
    """)

def display_theory_noscipy():
    st.subheader("Решение задания 12.2")
    st.write("""
    ##### Пример решения
    """)
    
    st.code("""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class ParabolicSolver2D:
    def __init__(self, l1, l2, T, N1, N2, M):
        #Инициализация параметров задачи
        #l1, l2 - размеры прямоугольника
        #T - конечное время
        #N1, N2 - количество точек по пространству
        #M - количество точек по времени
        
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
        #Точное решение задачи
        return t * x1 * (self.l1 - x1) * x2 * (self.l2 - x2)

    def initial_condition(self, x1, x2):
        #Начальное условие
        return 0.0

    def f(self, x1, x2, t):
        #Правая часть уравнения
        return (x1 * (self.l1 - x1) * x2 * (self.l2 - x2) +
                t * (-2 * x2 * (self.l2 - x2) - 2 * x1 * (self.l1 - x1)))

    def solve(self):
        #Решение задачи методом переменных направлений
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
        #Вычисление максимальной абсолютной погрешности
        error = 0.0
        for n in range(self.M + 1):
            for i in range(self.N1 + 1):
                for j in range(self.N2 + 1):
                    exact = self.exact_solution(self.x1[i], self.x2[j], self.t[n])
                    error = max(error, abs(self.u[n, i, j] - exact))
        return error

    def plot_solution(self, time_index):
        #Построение графика решения в заданный момент времени
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
        plt.show()


def main():
    # Параметры задачи
    l1 = l2 = 1.0  # размеры прямоугольника
    T = 1.0  # конечное время

    try:
        print("\nВведите параметры сетки:")
        N = int(input("Количество точек по пространству (N): "))
        M = int(input("Количество точек по времени (M): "))

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
            """)
    

def display_theory_scipy():
    st.subheader("Решение задания 12.2 с использованием библиотеки Scipy")
    st.write("""
    ##### Пример решения
    """)
    st.code("""
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from matplotlib import cm

# Параметры задачи
l1 = 1.0
l2 = 1.0
T = 1.1

N1 = 20
N2 = 20
M = 100

h1 = l1 / N1
h2 = l2 / N2
tau = T / M

x1 = np.linspace(0, l1, N1+1)
x2 = np.linspace(0, l2, N2+1)
t_grid = np.linspace(0, T, M+1)

# Определим точное решение и функцию f
def h1_fun(x):
    return x*(l1 - x)

def h2_fun(y):
    return y*(l2 - y)

def u_exact(i,j,n):
    return t_grid[n]*h1_fun(x1[i])*h2_fun(x2[j])

def f_func(i,j,n):
    return h1_fun(x1[i])*h2_fun(x2[j]) + 2*t_grid[n]*(h1_fun(x1[i]) + h2_fun(x2[j]))

# Начальное условие
U = np.zeros((N1+1, N2+1))
for i in range(N1+1):
    for j in range(N2+1):
        U[i,j] = u_exact(i,j,0)

# Коэффициенты для схемы Письмена-Рэчфорда
rx = tau/(2*h1*h1)
ry = tau/(2*h2*h2)

diag_x = (1 + 2*rx)*np.ones(N1-1)
off_x = -rx*np.ones(N1-2)
A_x = diags([off_x, diag_x, off_x], [-1,0,1], format='csr')

diag_y = (1 + 2*ry)*np.ones(N2-1)
off_y = -ry*np.ones(N2-2)
A_y = diags([off_y, diag_y, off_y], [-1,0,1], format='csr')

U_new = np.copy(U)
for n in range(M):
    # Промежуточный шаг по x
    U_star = np.zeros((N1+1, N2+1))
    for j in range(1, N2):
        RHS = np.zeros(N1-1)
        for i in range(1, N1):
            f_avg = 0.5*(f_func(i,j,n) + f_func(i,j,n+1))
            u_yy_n = (U[i,j+1] - 2*U[i,j] + U[i,j-1])/(h2*h2)
            RHS[i-1] = U[i,j] + (tau/2)*(u_yy_n + f_avg)
        U_star_inner = spsolve(A_x, RHS)
        for i in range(1, N1):
            U_star[i,j] = U_star_inner[i-1]
    U_star[0,:] = 0
    U_star[N1,:] = 0
    U_star[:,0] = 0
    U_star[:,N2] = 0

    # Основной шаг по y
    for i in range(1, N1):
        RHS = np.zeros(N2-1)
        for j in range(1, N2):
            f_avg = 0.5*(f_func(i,j,n) + f_func(i,j,n+1))
            u_xx_star = (U_star[i+1,j] - 2*U_star[i,j] + U_star[i-1,j])/(h1*h1)
            RHS[j-1] = U_star[i,j] + (tau/2)*(u_xx_star + f_avg)
        U_n1_inner = spsolve(A_y, RHS)
        for j in range(1, N2):
            U_new[i,j] = U_n1_inner[j-1]
    U_new[0,:] = 0
    U_new[N1,:] = 0
    U_new[:,0] = 0
    U_new[:,N2] = 0

    U = U_new.copy()

# Сравним с точным решением в момент времени T
U_exact_final = np.zeros((N1+1, N2+1))
for i in range(N1+1):
    for j in range(N2+1):
        U_exact_final[i,j] = u_exact(i,j,M)

err = np.max(np.abs(U - U_exact_final))
print("Max error = ", err)

# Визуализация результата:
X, Y = np.meshgrid(x2, x1)  # X соответствует второму направлению, Y - первому, чтобы была стандартная ориентация
# Отобразим численное решение:
fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(1, 2, 1, projection='3d')
surf = ax.plot_surface(X, Y, U, cmap=cm.viridis)
ax.set_title("Численное решение в t=T")
ax.set_xlabel('x2')
ax.set_ylabel('x1')
ax.set_zlabel('u')

# Отобразим точное решение:
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
surf2 = ax2.plot_surface(X, Y, U_exact_final, cmap=cm.viridis)
ax2.set_title("Точное решение в t={T}")
ax2.set_xlabel('x2')
ax2.set_ylabel('x1')
ax2.set_zlabel('u')

plt.tight_layout()
plt.show()

# Можно также отобразить разность:
fig2 = plt.figure()
ax3 = fig2.add_subplot(111, projection='3d')
surf3 = ax3.plot_surface(X, Y, U - U_exact_final, cmap=cm.coolwarm)
ax3.set_title("Разность (числ - точн) в t=T")
ax3.set_xlabel('x2')
ax3.set_ylabel('x1')
ax3.set_zlabel('U_num - U_exact')
plt.show()

            """)
    


def display_task_working():
    st.title("Решение двумерной задачи методом Письмена-Рэчфорда")
    st.subheader("C помощью библиотеки Scipy")
    # Параметры задачи
    l1 = 1.0
    l2 = 1.0
    T = 0.1

    # Настройки сетки через сайдбар
    N1 = st.sidebar.slider("Число разбиений по x1 (N1):", min_value=10, max_value=50, value=20)
    N2 = st.sidebar.slider("Число разбиений по x2 (N2):", min_value=10, max_value=50, value=20)
    M = st.sidebar.slider("Число шагов по времени (t):", min_value=50, max_value=200, value=100)

    h1 = l1 / N1
    h2 = l2 / N2
    tau = T / M

    x1 = np.linspace(0, l1, N1+1)
    x2 = np.linspace(0, l2, N2+1)
    t_grid = np.linspace(0, T, M+1)

    def h1_fun(x):
        return x*(l1 - x)

    def h2_fun(y):
        return y*(l2 - y)

    def u_exact(i,j,n):
        return t_grid[n]*h1_fun(x1[i])*h2_fun(x2[j])

    def f_func(i,j,n):
        return h1_fun(x1[i])*h2_fun(x2[j]) + 2*t_grid[n]*(h1_fun(x1[i]) + h2_fun(x2[j]))

    # Начальное условие
    U = np.zeros((N1+1, N2+1))
    for i in range(N1+1):
        for j in range(N2+1):
            U[i,j] = u_exact(i,j,0)

    # Коэффициенты для схемы Письмена-Рэчфорда
    rx = tau/(2*h1*h1)
    ry = tau/(2*h2*h2)

    diag_x = (1 + 2*rx)*np.ones(N1-1)
    off_x = -rx*np.ones(N1-2)
    A_x = diags([off_x, diag_x, off_x], [-1,0,1], format='csr')

    diag_y = (1 + 2*ry)*np.ones(N2-1)
    off_y = -ry*np.ones(N2-2)
    A_y = diags([off_y, diag_y, off_y], [-1,0,1], format='csr')

    U_new = np.copy(U)

    # Список для хранения решений на каждом шаге по времени
    solutions = [np.copy(U)]

    progress_bar = st.progress(0)
    for n in range(M):
        # Промежуточный шаг по x (U_star)
        U_star = np.zeros((N1+1, N2+1))
        for j in range(1, N2):
            RHS = np.zeros(N1-1)
            for i in range(1, N1):
                f_avg = 0.5*(f_func(i,j,n) + f_func(i,j,n+1))
                u_yy_n = (U[i,j+1] - 2*U[i,j] + U[i,j-1])/(h2*h2)
                RHS[i-1] = U[i,j] + (tau/2)*(u_yy_n + f_avg)
            U_star_inner = spsolve(A_x, RHS)
            for i in range(1, N1):
                U_star[i,j] = U_star_inner[i-1]
        U_star[0,:] = 0
        U_star[N1,:] = 0
        U_star[:,0] = 0
        U_star[:,N2] = 0

        # Основной шаг по y
        for i in range(1, N1):
            RHS = np.zeros(N2-1)
            for j in range(1, N2):
                f_avg = 0.5*(f_func(i,j,n) + f_func(i,j,n+1))
                u_xx_star = (U_star[i+1,j] - 2*U_star[i,j] + U_star[i-1,j])/(h1*h1)
                RHS[j-1] = U_star[i,j] + (tau/2)*(u_xx_star + f_avg)
            U_n1_inner = spsolve(A_y, RHS)
            for j in range(1, N2):
                U_new[i,j] = U_n1_inner[j-1]
        U_new[0,:] = 0
        U_new[N1,:] = 0
        U_new[:,0] = 0
        U_new[:,N2] = 0

        U = U_new.copy()
        solutions.append(np.copy(U))
        progress_bar.progress((n+1)/M)

    # Точное решение для всех моментов времени
    U_exact_all = []
    for n in range(M+1):
        Ue = np.zeros((N1+1, N2+1))
        for i in range(N1+1):
            for j in range(N2+1):
                Ue[i,j] = u_exact(i,j,n)
        U_exact_all.append(Ue)

    # Выбор момента времени для визуализации
    time_step = st.slider("Выберите номер шага по времени для визуализации", 0, M, 0)
    U_num_show = solutions[time_step]
    U_exact_show = U_exact_all[time_step]
    X, Y = np.meshgrid(x2, x1)

    # Построение графиков
    fig = plt.figure(figsize=(12,4))

    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax1.plot_surface(X, Y, U_num_show, cmap=cm.viridis)
    ax1.set_title(f"$Численное\;решение\;(t={t_grid[time_step]:.3f})$")
    ax1.set_xlabel('$x2$')
    ax1.set_ylabel('$x1$')

    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax2.plot_surface(X, Y, U_exact_show, cmap=cm.viridis)
    ax2.set_title(f"$Точное\;решение\;(t={t_grid[time_step]:.3f})$")
    ax2.set_xlabel('$x2$')
    ax2.set_ylabel('$x1$')

    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    ax3.plot_surface(X, Y, U_num_show - U_exact_show, cmap=cm.coolwarm)
    ax3.set_title("$Разность\;численного\;решения\;и\;точного$")
    ax3.set_xlabel('$x2$')
    ax3.set_ylabel('$x1$')

    st.pyplot(fig)

    # Вывод ошибки для текущего шага по времени
    err_current = np.max(np.abs(U_num_show - U_exact_show))
    st.write(f"Максимальная ошибка на выбранном шаге по времени: {err_current}")

    printme1, res1 = countexample_metod1()
    
    st.write("# Решение без scipy")
    st.write(printme1)


def countexample_metod1():
    f = io.StringIO()
    with redirect_stdout(f):
        res = p12_2.main()
    output_string = f.getvalue()
    return output_string, res



def display_extra():
    st.subheader("Дополнительная информация о задании 12.2")
    st.write("""
    Программа решатель 
    
    - Скорых Александр
    
    Программа решатель с SciPy
    
    - Тищенко Илья
    
    Презентация в streamlit
    
    - Хаметов Марк
    
    """)
    #st.image("logo.png")
