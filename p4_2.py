

#--------------------------------------------------------------------------------------------------------
#"""""""""""""""""""""""""""""""""""""""""""""""" 4.2 c scipy """"""""""""""""""""""""""""""""""""""""""""
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

    print("Решение для n = ", n)
    x0 = np.ones(n)  # Начальное приближение
    result = root(F, x0, method='hybr')  # Можно также использовать 'lm', 'broyden1' и другие методы
    if result.success:
        print("Найденное решение:`", result.x)
        print("`")
        return result.x
        
    else:
        raise ValueError("SciPy не нашел решение")

# Проверка на примере n = 5
n = 5
solution_scipy = scipy_solver(n)
print("Решение с SciPy:", solution_scipy)
