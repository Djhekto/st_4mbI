import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import io
from contextlib import redirect_stdout


import p12_1
import pypy


def run_task12_1():
    
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
    st.subheader("Задание 12.1")
    st.write("""
##### 12 Нестационарные задачи математической физики

**Задание 12.1**

Напишите программу численного решения на равномерной
по пространству и времени сетке задачи

$\\begin{aligned}
  \\frac{\partial u}{\partial t}=
  \\frac\partial{\partial x}\left(k(x)\\frac{\partial u}{\partial
  x}\\right) + f(x,t) ,
  \quad 0 < x < l,
  \quad 0 < t \le T ,
\end{aligned}$

$\\begin{aligned}
  u(0,t)=0,
  \quad u(l,t)=0,
  \quad 0 < t \le T ,
\end{aligned}$

$\\begin{aligned}
u(x,0)=v(x), \quad 0 \le x \le l
\end{aligned}$

при использовании двухслойной схемы с весом $\sigma$.
С помощью этой программы проведите численные эксперименты
по приближенному решению задачи с $k(x) = 1$, $f(x,t) = 0$
и точным решением ($l=1$)

$\\begin{aligned}
  u(x,t) = e^{-\pi^2 t} \sin(\pi x) + e^{-16\pi^2 t} \sin(4\pi x)
\end{aligned}$

на различных сетках по времени при использовании симметричной
($\sigma = 0.5$) и чисто неявной ($\sigma = 1$) разностных схем.

Решите также эту задачу с помощью библиотеки SciPy.
    """)

def display_theory():
    st.subheader("О методе бисекций")
    st.write(f"""
    ##### Уравнения математической физики

    Рассмотрим уравнение теплопроводности с переменным коэффициентом теплопроводности \( k(x) \) и внешней силой \( f(x, t) \):
    """)
    
    st.latex("""

\\frac{\partial u}{\partial t} = \\frac{\partial}{\partial x} \left( k(x) \\frac{\partial u}{\partial x} \\right) + f(x, t),

\quad 0 

где  u(x, t)  — это температура в точке  x  и времени  t ,  k(x)  — коэффициент, зависящий от положения.
             
             """)

    st.write(f"""

### Шаги для анализа и решения задачи:

- **Физический смысл**:

- Уравнение описывает, как температура  u  изменяется во времени под действием как внутренней диффузии
    за счет
    """)
    
    st.latex("""
     k(x) \\frac{\\partial u}{\partial x} ), так и внешнего источника или sink  f(x,t) .
    """)

    st.write(f"""
**Граничные условия**:

- Настройте условия на границах \( x = 0 \) и \( x = l \). Например, это могут быть абсолютные условия, такие как:

- \( u(0, t) = u_0(t) \) (постоянная температура на одном конце);

- \( u(l, t) = u_l(t) \) (постоянная температура на другом конце);

- Начальное условие: \( u(x, 0) = u_i(x) \) – начальное распределение температуры.

**Волновая и диффузионная части**:

- Уравнение можно разделить на одну часть, связанную с пространственным распределением, и другую часть со временнóй изменчивостью.

**Методы решения**:

- Для решения этого PDE можно использовать метод разделения переменных, где предполагается, что \( u(x, t) = X(x)T(t) \). После подстановки и деления на \( XT \), получится два обыкновенных дифференциальных уравнения, одно для \( X(x) \) и одно для \( T(t) \), которые можно решать по отдельности.

- Либо можно воспользоваться численными методами (например, методом конечных разностей) для численного интегрирования в дегрессивной области.

**Специальные случаи**:

- Если \( k(x) \) является постоянным, уравнение превращается в классическое уравнение теплопроводности. В этом случае можно использовать стандартные решения (функции Грина и т.д.).

- Если \( f(x, t) = 0 \), оно сводится к гомогенной модели, где анализируются только теплопроводность и начальные/границы. 

**Анализ решений**:

- После решения можно проанализировать поведение температуры в разное время (например, с помощью графиков или численных таблиц), что будет важным для понимания всех динамических процессов.

Таким образом, это уравнение описывает сложные процессы, которые могут происходить в материале с учетом внешних воздействий, что делает его значительным для многих реальных физических приложений.

    """)

def display_theory_noscipy():
    st.subheader("Решение задания 12.1")
    st.write("""
    ##### Пример кода
    """)
    
    st.code("""
import numpy as np
import matplotlib.pyplot as plt
from math import pi, exp, sin


def k(x):
    #Коэффициент теплопроводности
    return 1.0


def f(x, t):
    #Правая часть уравнения
    return (-pi ** 2 * exp(-pi ** 2 * t) * sin(pi * x) +
            -16 * pi ** 2 * exp(-16 * pi ** 2 * t) * sin(4 * pi * x))


def v(x):
    #Начальное условие 
    return sin(pi * x) + sin(4 * pi * x)


def exact_solution(x, t):
    #Точное решение 
    return exp(-pi ** 2 * t) * sin(pi * x) + exp(-16 * pi ** 2 * t) * sin(4 * pi * x)


def solve_parabolic(N, M, T, L, q):
    #Решение параболической задачи
    #N - число точек по пространству
    #M - число точек по времени
    #T - конечное время
    #L - длина отрезка
    #q - вес схемы

    # Шаги сетки
    h = L / N  # по пространству
    tau = T / M  # по времени

    # Создание сеток
    x = np.linspace(0, L, N + 1)
    t = np.linspace(0, T, M + 1)

    # Создание массива решения
    u = np.zeros((M + 1, N + 1))

    # Начальное условие
    for i in range(N + 1):
        u[0, i] = v(x[i])

    # Граничные условия
    for j in range(M + 1):
        u[j, 0] = 0
        u[j, N] = 0

    # Коэффициенты для матрицы системы
    r = tau / (h ** 2)

    # Формирование матрицы системы
    A = np.zeros((N - 1, N - 1))
    for i in range(N - 1):
        if i > 0:
            A[i, i - 1] = -q * r
        A[i, i] = 1 + 2 * q * r
        if i < N - 2:
            A[i, i + 1] = -q * r

    # Решение по временным слоям
    for j in range(M):
        # Формирование правой части
        b = np.zeros(N - 1)
        for i in range(N - 1):
            b[i] = u[j, i + 1] + tau * ((1 - q) * f(x[i + 1], t[j]) + q * f(x[i + 1], t[j + 1]))
            if i > 0:
                b[i] += (1 - q) * r * u[j, i]
            if i < N - 2:
                b[i] += (1 - q) * r * u[j, i + 2]
            b[i] += (1 - q) * r * (-2 * u[j, i + 1])

        # Решение СЛАУ
        u[j + 1, 1:N] = np.linalg.solve(A, b)

    return x, t, u


def calculate_error(u_num, u_exact, x, t):
    #Вычисление максимальной абсолютной погрешности
    error = 0.0
    for i in range(len(t)):
        for j in range(len(x)):
            error = max(error, abs(u_num[i, j] - u_exact(x[j], t[i])))
    return error


def plot_results(x, t, u_num, title):
    #Построение графика численного решения
    X, T = np.meshgrid(x, t)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, T, u_num, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('u')
    ax.set_title(title)
    plt.show()


def main():
    # Параметры задачи
    L = 1.0  # длина отрезка
    T = 0.1  # конечное время

    # Списки параметров для экспериментов
    N_values = [20, 40, 80]  # точки по пространству
    M_values = [20, 40, 80]  # точки по времени
    q_values = [0.5, 1.0]  # веса схемы

    for q in q_values:
        print(f"\nВес схемы q = {q}")
        print("\nТаблица погрешностей:")
        print("   N    M      Погрешность")
        print("-" * 30)

        for N in N_values:
            for M in M_values:
                # Решение задачи
                x, t, u_num = solve_parabolic(N, M, T, L, q)

                # Вычисление погрешности
                error = calculate_error(u_num, exact_solution, x, t)

                print(f"{N:4d} {M:4d} {error:15.2e}")

                # Построение графика для некоторых параметров
                if N == 40 and M == 40:
                    title = f"Численное решение (q={q}, N={N}, M={M})"
                    plot_results(x, t, u_num, title)

                    # Построение точного решения для сравнения
                    u_exact = np.zeros((M + 1, N + 1))
                    for i in range(M + 1):
                        for j in range(N + 1):
                            u_exact[i, j] = exact_solution(x[j], t[i])
                    plot_results(x, t, u_exact, f"Точное решение (N={N}, M={M})")


if __name__ == "__main__":
    main()
            """)
    

def display_theory_scipy():
    st.subheader("Решение задания 12.1")
    st.write("""
    ##### Пример кода
    """)
    
    st.code("""
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# Параметры задачи
l = 1
T = 1
Nx = 35  # Число точек по пространству
Nt = 1000  # Число точек по времени
sigma = 0.5  # Симметричная схема

dx = l / Nx
dt = T / Nt
x = np.linspace(0, l, Nx + 1)
t = np.linspace(0, T, Nt + 1)

# Функция точного решения
def exact_solution(x, t):
    return np.exp(-np.pi**2 * t) * np.sin(np.pi * x) + np.exp(-16 * np.pi**2 * t) * np.sin(4 * np.pi * x)


# Решение с использованием SciPy
u_scipy = np.zeros((Nt + 1, Nx + 1))
u_scipy[0, :] = exact_solution(x, 0)

alpha = sigma * dt / dx**2
diagonal = (1 + 2 * alpha) * np.ones(Nx - 1)
off_diagonal = -alpha * np.ones(Nx - 2)
A = diags([off_diagonal, diagonal, off_diagonal], [-1, 0, 1]).tocsc()

for n in range(0, Nt):
    rhs = u_scipy[n, 1:-1] + (1 - sigma) * (u_scipy[n, 2:] - 2 * u_scipy[n, 1:-1] + u_scipy[n, :-2]) * dt / dx**2
    u_scipy[n + 1, 1:-1] = spsolve(A, rhs)

# ---------------------------------------------
# Построение графиков
plt.figure(figsize=(14, 7))

# График для метода с использованием SciPy
for i in range(0, Nt + 1, Nt // 5):
    plt.plot(x, u_scipy[i, :], label=f'$t={t[i]:.2f}$')
plt.plot(x, exact_solution(x, T), 'k--', label='$Точное\;решение$')
plt.title('$Решение\;с\;SciPy$')
plt.xlabel('$x$')
plt.ylabel('$u$')
plt.legend()

plt.tight_layout()
plt.show()

            """)
    
def display_task_working():
    st.title("Численное решение нестационарной задачи математической физики")

    st.write(r"""
    Решаем уравнение: 
    $$
    \frac{\partial u}{\partial t} = \frac{\partial}{\partial x}\left(k(x)\frac{\partial u}{\partial x}\right) + f(x,t), \quad 0 < x < l, \quad 0 < t \leq T
    $$

    Граничные условия:
    $$
    \begin{aligned}
      u(0,t) &= 0, \\
      u(l,t) &= 0, \quad 0 < t \leq T
    \end{aligned}
    $$

    Начальное условие:
    $$
    \begin{aligned}
      u(x,0) = v(x), \quad 0 \leq x \leq l
    \end{aligned}
    $$
    """)
    st.subheader("C использованием библиотеки Scipy:")

    # Параметры задачи с возможностью настройки
    st.sidebar.header("Настройки задачи")
    l = 1
    T = 1
    Nx = st.slider("Число точек по пространству (Nx)", 10, 100, 30)
    Nt = st.slider("Число точек по времени (Nt)", 100, 2000, 1000)
    sigma = st.slider("Симметричная схема (sigma)", 0.1, 1.0, 0.5)

    dx = l / Nx
    dt = T / Nt
    x = np.linspace(0, l, Nx + 1)
    t = np.linspace(0, T, Nt + 1)

    # Функция точного решения
    def exact_solution(x, t):
        return np.exp(-np.pi**2 * t) * np.sin(np.pi * x) + np.exp(-16 * np.pi**2 * t) * np.sin(4 * np.pi * x)

    # Решение с использованием SciPy
    u_scipy = np.zeros((Nt + 1, Nx + 1))
    u_scipy[0, :] = exact_solution(x, 0)

    alpha = sigma * dt / dx**2
    diagonal = (1 + 2 * alpha) * np.ones(Nx - 1)
    off_diagonal = -alpha * np.ones(Nx - 2)
    A = diags([off_diagonal, diagonal, off_diagonal], [-1, 0, 1]).tocsc()

    for n in range(0, Nt):
        rhs = u_scipy[n, 1:-1] + (1 - sigma) * (u_scipy[n, 2:] - 2 * u_scipy[n, 1:-1] + u_scipy[n, :-2]) * dt / dx**2
        u_scipy[n + 1, 1:-1] = spsolve(A, rhs)

    # Построение графиков
    st.header("Решение:")

    # График для метода с использованием SciPy
    st.subheader("Решение с использованием SciPy")
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    for i in range(0, Nt + 1, Nt // 5):
        ax2.plot(x, u_scipy[i, :], label=f'$t={t[i]:.2f}$')
    ax2.plot(x, exact_solution(x, T), 'k--', label='$Точное\;решение$')
    ax2.set_title('$Решение\;с\;SciPy$')
    ax2.set_xlabel('$x$')
    ax2.set_ylabel('$u$')
    ax2.legend()
    st.pyplot(fig2)

    st.write("# Решение без Scipy")
    
    printme1, res1  = countexample_metod1()
    st.write(printme1)

def countexample_metod1():
    f = io.StringIO()
    with redirect_stdout(f):
        res = p12_1.main()
    output_string = f.getvalue()
    return output_string, res



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
