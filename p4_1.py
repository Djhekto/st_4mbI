#"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#""""""""""""""""""""""""""""""""""""""""""      4.1 c scipy      """"""""""""""""""""""""""""""""""""""""
#"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import numpy as np
import scipy

# Определяем функцию
def f(x):
    return (1 + x**2) * np.exp(-x) + np.sin(x)

# Основной интервал поиска
def mainfunc(a,b,e,iterc):

    # Генерация точек для анализа функции
    x_vals = np.linspace(a, b, iterc)
    y_vals = f(x_vals)

    # Поиск интервалов смены знака
    sign_changes = np.where(np.sign(y_vals[:-1]) != np.sign(y_vals[1:]))[0]
    root_intervals = [(x_vals[i], x_vals[i+1]) for i in sign_changes]

    # Поиск корней на найденных интервалах методом бисекции
    roots = []
    for interval in root_intervals:
        result = scipy.optimize.root_scalar(f, method='bisect', bracket=interval)
        if result.converged:
            roots.append(round(result.root, 6))

    # Вывод найденных корней
    lroots  = []
    print("Найдены корни уравнения:")
    for i, root in enumerate(roots, 1):
        lroots.append(root)
        print(f"Корень {i}: x = {root:}")
    
    return lroots
