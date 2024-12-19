import numpy as np


def kernel(x, s):
    """Ядро интегрального уравнения: |x-s|"""
    return abs(x - s)


def right_side(x):
    """Правая часть уравнения: x^2"""
    return x ** 2


def exact_solution(x):
    """Точное решение уравнения: y(x) = 3x"""
    return 3 * x


def conjugate_gradient_method(A, b, tol):
    """
    Метод сопряженных градиентов для решения системы Ax = b
    A - симметричная матрица
    b - вектор правой части
    tol - требуемая точность
    """
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
    """
    Решение интегрального уравнения методом квадратур
    n - количество точек разбиения
    tol - требуемая точность решения
    """
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
    """Вычисление максимальной абсолютной погрешности"""
    return np.max(np.abs(y_approx - y_exact))


def main11(n, tol = 1e-6):
    try:
        # Ввод параметров
        #n = int(input("Введите количество точек разбиения (n > 1): "))


        #tol = float(input("Введите требуемую точность (например, 1e-6): "))


        # Решение уравнения
        x, y_approx, iterations = solve_fredholm(n, tol)

        # Вычисление точного решения
        y_exact = exact_solution(x)

        # Вычисление погрешности
        error = calculate_error(x, y_approx, y_exact)

        # Вывод результатов
        print("\nРезультаты:")
        print(f"Количество итераций: {iterations}")
        print(f"Максимальная абсолютная погрешность: {error:.2e}\n")
        print("\nСравнение решений:")
        print("    x      Приближенное  Точное")
        print("-" * 40)
        for i in range(n):
            print(f"{x[i]:7.3f} {y_approx[i]:12.6f} {y_exact[i]:12.6f}\n")
        
        return y_approx, x


    except ValueError:
        print("Ошибка ввода! Проверьте формат данных.")
    except np.linalg.LinAlgError:
        print("Ошибка! Не удалось решить систему уравнений.")
    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")


#if __name__ == "__main__":
#    main()