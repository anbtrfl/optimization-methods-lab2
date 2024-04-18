import math
import tracemalloc
from time import time

import numpy as np
import sympy
from matplotlib import pyplot as plt
from numpy import float64
from scipy import optimize
from sympy import *


class Statistic:
    def __init__(self):
        self.__start_time = None
        self.iterations = 0
        self.spent_time = 0.0
        self.memory = 0
        self.function_calculations = 0
        self.gradient_calculations = 0
        self.hessian_calculations = 0
        self.is_tracing_running = False

    def start_clock(self):
        self.__start_time = time()

    def stop_clock(self):
        self.spent_time += time() - self.__start_time
        self.__start_time = None

    def start_trace(self):
        tracemalloc.start()
        self.is_tracing_running = True

    def stop_trace(self):
        if self.is_tracing_running:
            res = tracemalloc.take_snapshot()
            stats = res.statistics(cumulative=True, key_type='filename')
            for stat in stats:
                self.memory += stat.size
            tracemalloc.stop()
            self.is_tracing_running = False

    def print_stat(self):
        print('time:           ', self.spent_time)
        print('memory:         ', self.memory)
        print('function_calls: ', self.function_calculations)
        print('gradient_calls: ', self.gradient_calculations)
        print('hessian_calls:  ', self.hessian_calculations)
        print('iterations:     ', self.iterations)


def difficult_func(func, vars, name):
    dfdx = lambdify(vars, diff(func, vars[0]))
    dfdy = lambdify(vars, diff(func, vars[1]))

    hess = hessian(func, vars)
    print(hess)
    hess = lambdify(vars, hess, modules='sympy')
    func = lambdify(vars, func, 'numpy')

    def f(args):
        return func(args[0], args[1])

    def grad(x):
        return np.array([dfdx(x[0], x[1]), dfdy(x[0], x[1])], dtype=float64)

    def f_hessian(x):
        res = hess(x[0], x[1])
        result = np.array([res[:2], res[2:]], dtype=float64)
        return result

    print(f_hessian(np.array([0, 0])))
    return [f, grad, f_hessian, name]


def not_working():
    x, y = symbols('x, y')
    func = 0.1 * (x ** 4 + y ** 4) + x * y ** 2 + 2 * x + 2 * y
    return difficult_func(func, np.array([x, y]), "func")


def distribution_function():
    x, y = symbols('x, y')
    func = 5 * (2 * x ** 2 - 1) * y * sympy.exp(-x ** 2 - y ** 2)
    return difficult_func(func, np.array([x, y]),
                          "5 * (2x^2 - 1)y * exp(-x^2-y^2)")


def rosenbrock(x):
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


def grad_rosenbrock(arg):
    x = arg[0]
    y = arg[1]
    dx = 2 * (200 * (x ** 3) - 200 * x * y + x - 1)
    dy = 200 * (y - (x ** 2))
    return np.array([dx, dy])


def hessian_rosenbrock(arg):
    x = arg[0]
    y = arg[1]
    return np.array([
        np.array([-400 * (y - x ** 2) + 800 * x ** 2 + 2, -400 * x]),
        np.array([-400 * x, 200])
    ])


def ackley():
    x, y = symbols('x, y')
    func = -20.0 * exp(-0.2 * sqrt(0.5 * (x ** 2 + y ** 2))) - exp(
        0.5 * (cos(2 * pi * x) + cos(2 * pi * x))) + math.e + 20
    return difficult_func(func, np.array([x, y]),
                          "ackley")


def notpolinom():
    x, y = symbols('x, y')
    func = 5 * x * y * exp(-x ** 2 - y ** 2)
    return difficult_func(func, np.array([x, y]),
                          "5 * x * y * exp(-x ** 2 - y ** 2)")


def reversed_f():
    x, y = symbols('x, y')
    func = -1 * (2 * exp(-((x - 1) / 2) ** 2 - ((y - 1) / 1) ** 2) + 3 * exp(
        -((x - 2) / 3) ** 2 - ((y - 3) / 2) ** 2))
    return difficult_func(func, np.array([x, y]), "notpolinom2")


def golden_ratio(function, left_border, right_border, x, p, stat, eps=1e-8):
    phi = (1 + np.sqrt(5)) / 2
    a = left_border
    b = right_border
    iterations = 0
    calculations = 0
    segments = [(a, b)]

    c_1 = b - (b - a) / phi
    c_2 = a + (b - a) / phi

    stat.function_calculations += 2
    calc_result = [function(x + p * c_1), function(x + p * c_2)]
    calculations += 2

    while (b - a) / 2 >= eps:
        iterations += 1
        calculations += 1
        stat.function_calculations += 1
        if calc_result[0] > calc_result[1]:
            a = c_1
            c_1 = c_2
            c_2 = b - (c_1 - a)
            calc_result[0] = calc_result[1]
            calc_result[1] = function(x + p * c_2)
        else:
            b = c_2
            c_2 = c_1
            c_1 = a + b - c_2
            calc_result[1] = calc_result[0]
            calc_result[0] = function(x + p * c_1)
        segments.append((a, b))

    c = (b + a) / 2

    return c


def dichotomies(f, a, b, stat, eps=1e-8):
    while abs(b - a) > eps:
        c = (a + b) / 2
        delta = (b - a) / 8
        f1 = f(c - delta)
        f2 = f(c + delta)
        stat.function_calculations += 2
        if f1 < f2:
            b = c
        else:
            a = c
    return (a + b) / 2


def gradient_descend(f, f_grad, hessian, start_point, eps=1e-8, epochs=1100):
    stat = Statistic()
    stat.start_trace()
    stat.start_clock()
    history = [start_point]
    x = start_point
    for epoch in range(1, epochs + 1):
        grad_value = f_grad(x)
        lr = dichotomies(lambda t: f(x - t * grad_value), 0, 2, stat)
        x = x - lr * grad_value
        history.append(x)
        if epoch > 0 and np.linalg.norm(f_grad(x)) < eps:
            break
        stat.gradient_calculations += 1
        stat.iterations += 1
    stat.stop_trace()
    stat.stop_clock()
    return history[-1], stat, history


def positive_matrix(m):
    eigenvalues = np.linalg.eigvals(m)
    if all(eigenvalues > 0):
        return m
    else:
        return m + 2 * (-min(eigenvalues)) * np.eye(m.shape[0])


def newton(f, f_grad, f_hessian, start_point, eps=1e-8):
    cur_x = start_point
    stat = Statistic()
    stat.start_trace()
    stat.start_clock()
    xs = [start_point]
    cur_grad = 1

    while stat.iterations < 1000 and np.linalg.norm(cur_grad) > eps:
        xs.append(cur_x)
        stat.iterations += 1
        cur_hessian = f_hessian(cur_x)
        stat.hessian_calculations += 1
        pos_hessian = positive_matrix(cur_hessian)
        cur_grad = f_grad(cur_x)
        if (np.linalg.det(pos_hessian) != 0):
            direction = np.linalg.inv(pos_hessian) @ cur_grad
        else:
            direction = cur_grad
        alpha = golden_ratio(f, 0, 5, cur_x, -direction, stat)
        cur_x = cur_x - direction * alpha
        stat.gradient_calculations += 1
    stat.stop_trace()
    stat.stop_clock()
    xs.append(cur_x)
    return cur_x, stat, xs


def newton_with_constant_step(f, f_grad, f_hessian, start_point, eps=1e-8, step=0.1):
    cur_x = start_point
    stat = Statistic()
    stat.start_trace()
    stat.start_clock()
    xs = [start_point]
    cur_grad = 1

    while stat.iterations < 1000 and np.linalg.norm(cur_grad) > eps:
        xs.append(cur_x)
        stat.iterations += 1
        cur_hessian = f_hessian(cur_x)
        stat.hessian_calculations += 1
        pos_hessian = positive_matrix(cur_hessian)
        cur_grad = f_grad(cur_x)
        if np.linalg.det(pos_hessian) != 0 and stat.iterations > 3:
            direction = np.linalg.inv(pos_hessian) @ cur_grad
        else:
            print("zero")
            direction = cur_grad
        cur_x = cur_x - direction * step
        stat.gradient_calculations += 1
    stat.stop_trace()
    stat.stop_clock()
    xs.append(cur_x)
    return cur_x, stat, xs


def wolf_condition(c1, c2, grad, f, xk, ak, direction, stat):
    grad_xk = grad(xk)
    new_xk = xk + ak * direction
    armiho_condition = f(new_xk) <= f(xk) + c1 * ak * np.dot(grad_xk, direction)
    curvature_condition = abs(np.dot(grad(new_xk), direction)) <= c2 * abs(np.dot(grad_xk, direction))
    stat.function_calculations += 2
    stat.gradient_calculations += 2
    return (armiho_condition and curvature_condition) or ak < 1e-10


def backtracking_line_search(f, grad, a1, a2, xk, direction, stat):
    beta = 0.9
    c1 = 0.01
    c2 = 0.5
    ak = 1
    while not wolf_condition(c1, c2, grad, f, xk, ak, direction, stat):
        ak *= beta
    return ak


def newton_with_wolf(f, f_grad, f_hessian, start_point, eps=1e-8):
    cur_x = start_point
    stat = Statistic()
    stat.start_trace()
    stat.start_clock()
    xs = []
    cur_grad = f_grad(cur_x)
    while stat.iterations < 1000 and np.linalg.norm(cur_grad) > eps:
        xs.append(cur_x)
        stat.iterations += 1
        cur_hessian = f_hessian(cur_x)
        stat.hessian_calculations += 1
        pos_hessian = positive_matrix(cur_hessian)
        if np.linalg.det(pos_hessian) != 0 and stat.iterations > 3:
            direction = np.linalg.inv(pos_hessian) @ cur_grad
        else:
            print("zero")
            direction = cur_grad
        alpha = backtracking_line_search(f, f_grad, 0, 5, cur_x, -direction, stat)
        cur_x = cur_x - direction * alpha
        cur_grad = f_grad(cur_x)
        stat.gradient_calculations += 1
    stat.stop_trace()
    stat.stop_clock()
    xs.append(cur_x)
    return cur_x, stat, xs


def generic_quazi_optimize_function(f, f_grad, start_point, eps, method, f_hessian=None):
    stat = Statistic()
    stat.start_trace()
    stat.start_clock()
    xs = [start_point]
    result = optimize.minimize(fun=f, x0=start_point, method=method, jac=f_grad, hess=f_hessian, tol=eps,
                               callback=lambda el: xs.append(el),
                               options={'maxiter': 100000})
    stat.function_calculations = result.nfev
    stat.gradient_calculations = result.njev
    stat.iterations = result.nit
    stat.stop_trace()
    stat.stop_clock()
    return result.x, stat, xs


def newton_cg(f, f_grad, f_hessian, start_point, eps=1e-8):
    return generic_quazi_optimize_function(f, f_grad, start_point, eps, 'Newton-CG')


def quasinewton_BFGS(f, f_grad, f_hessian, start_point, eps=1e-8):
    return generic_quazi_optimize_function(f, f_grad, start_point, eps, 'BFGS')


def quasinewton_L_BFGS(f, f_grad, f_hessian, start_point, eps=1e-8):
    return generic_quazi_optimize_function(f, f_grad, start_point, eps, 'L-BFGS-B')


def print_result(x, f, stat):
    print('x: ', x)
    print('y: ', f(x))
    stat.print_stat()


all_functions = [

    [rosenbrock, grad_rosenbrock, hessian_rosenbrock, "Rosenbrock"],

    distribution_function(),

    not_working(),

]

start_points = [
    # для доп зад2 пункт 1 точки
    # np.array([0, -0.5]),
    # np.array([1.5, -1.2]),
    # np.array([0.4, 0])
    # np.array([11, 7], dtype=float64),

    # доп зад2 пункт2
    # np.array([0, 0], dtype=float64),

    # np.array([0, 0]),
    np.array([3, 5], dtype=float64),
    # np.array([-1, 5]),
    # np.array([-4, -4]),
    # np.array([-4, 4]),
    # np.array([4, 4]),
    # np.array([4, -4]),
    # np.array([16, 10], dtype=float64),
    # # np.array([5, 1]),
    # np.array([80, 80], dtype=float64),
    # np.array([200, 200], dtype=float64),
    # np.array([1000, 1000], dtype=float64),
]

methods = [
    [gradient_descend, "Gradient descend"],
    [newton, "Newton"],
    [newton_with_constant_step, "Newton with constant step"],
    [newton_with_wolf, "Newton with wolf"],
    [newton_cg, "Newton-CG"],
    [quasinewton_BFGS, "Quasinewton (BFGS)"],
    [quasinewton_L_BFGS, "Quasinewton (L-BFGS-B)"]
]

results_for_graphs = []
stats_by_method = {
    gradient_descend: [],
    newton: [],
    newton_with_constant_step: [],
    # метод ньютона с условием вольфа
    newton_with_wolf: [],
    newton_cg: [],
    quasinewton_BFGS: [],
    quasinewton_L_BFGS: []
}


def draw(diap, eps, function, start_values, title, xk, yk, fig, ax, ax2):
    x = np.arange(-diap[0], diap[0], diap[0] / 100)
    y = np.arange(-diap[1], diap[1], diap[1] / 100)
    X, Y = np.meshgrid(x, y)
    Z = function([X, Y])
    ax.plot(xk, yk, function(np.array([xk, yk])), color='blue')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, color='cyan', edgecolor='none', alpha=0.3)

    # линии уровня
    cp = ax2.contour(X, Y, Z, levels=sorted(set(function(np.array([xk[i], yk[i]])) for i in range(len(xk)))))
    ax2.clabel(cp, inline=1, fontsize=10)
    ax2.plot(xk, yk, color='blue')

    fig.suptitle(f"{title}, epsilon = {eps}, начальная точка: {start_values}")


def draw_result(result, function, title, eps):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2)
    xs = result[2]
    xk = [i[0] for i in xs]
    yk = [i[1] for i in xs]
    diapason = np.array([max(map(abs, xk)) + 3, max(map(abs, yk)) + 3])
    draw(diapason, eps, function, point, title, xk, yk, fig, ax, ax2)
    plt.show()


for function in all_functions:
    for point in start_points:
        print("=================================================================================")
        print("Function name: " + function[-1])
        print()
        for method in methods:
            x, stat, xs = method[0](function[0], function[1], function[2], point)
            print("Method: " + method[1])
            print("Start point: ", point)
            print_result(x, function[0], stat)
            print()
            results_for_graphs.append((point, method[-1], function[-1], xs, function[0]))
            stats_by_method[method[0]].append(stat)
            draw_result((x, stat, xs), function[0], method[-1] + " для функции " + function[-1], 1e-8)
        print()
        print()

point = np.array([2, 5])
newton_result = newton(rosenbrock, grad_rosenbrock, hessian_rosenbrock, point)
newton_result_constant_step = newton_with_constant_step(rosenbrock, grad_rosenbrock, hessian_rosenbrock, point,
                                                        step=0.1)
plt.show()
