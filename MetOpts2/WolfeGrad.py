import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (20, 10)


def fun(x, y):
    return np.sin(0.5 * x ** 2 - 0.25 * y ** 2 + 3) * np.cos(2 * x + 1 - np.exp(y))


def grad(x, y):
    h = 1e-5
    return [(fun(x + h, y) - fun(x - h, y)) / (2 * h), (fun(x, y + h) - fun(x, y - h)) / (2 * h)]


def gradSquare(x, y):
    h = 1e-5
    return ((fun(x + h, y) - fun(x - h, y)) / (2 * h)) ** 2 + ((fun(x, y + h) - fun(x, y - h)) / (2 * h)) ** 2


def calc_fun_by_t(x, y, grad1, t1):
    return fun(x - t1 * grad1[0], y - t1 * grad1[1])


def multVector(a, b):
    return a[0] * b[0] + a[1] * b[1]


def searchLearningRate(x, y, grad1):
    resLR = 0.5
    a = 0  # Левая граница t
    b = 1  # Правая граница t
    # находим отрезок, так что: f(a) < f(b)
    while calc_fun_by_t(x, y, grad1, a) >= calc_fun_by_t(x, y, grad1, b):
        if a == 0:
            a = 1
            b = 2
        else:
            a *= 2
            b *= 2
    a = 0
    c1 = 0.0001
    c2 = 0.9
    eps_for_t = 0.001
    while (calc_fun_by_t(x, y, grad1, resLR) > fun(x, y) - c1 * resLR * gradSquare(x, y)
           or multVector(grad(x - resLR * grad1[0], y - resLR * grad1[1]), grad1) < c2 * gradSquare(x, y) * -1):
        print(a, " ", b)
        # print("aboba")
        # Находим середину и две близких точки
        mid = (b + a) / 2
        t1 = mid - eps_for_t
        t2 = mid + eps_for_t
        f_t1 = calc_fun_by_t(x, y, grad1, t1)
        f_t2 = calc_fun_by_t(x, y, grad1, t2)
        # Сужаем отрезок
        if f_t1 < f_t2:
            b = t2
        else:
            if f_t1 > f_t2:
                a = t1
            else:
                resLR = a
        resLR = (a + b) / 2
    return resLR


eps = 0.001

pointMas = np.linspace(-1.5, 1.5, 100)
X, Y = np.meshgrid(pointMas, pointMas)

learningRate = 0.1
point = [-0.1, -0.4]

points = []
points.append(list(point))
i = 1
while True:
    learningRate = searchLearningRate(point[0], point[1], grad(point[0], point[1]))
    point -= learningRate * np.array(grad(point[0], point[1]))
    points.append(list(point))
    if abs(fun(points[i - 1][0], points[i - 1][1]) - fun(points[i][0], points[i][1])) < eps:
        break
    i += 1

figure = plt.figure()
ax1 = plt.figure().add_subplot()
ax = plt.figure().add_subplot(projection='3d')
ax.plot_surface(X, Y, fun(X, Y))
array = np.array(points)
ax1.plot(fun(array[:, 0], array[:, 1]))
ax1.grid()
print("Minimum: ", fun(array[-1, 0], array[-1, 1]))
print("Number of iterations: ", i)
print("Point: ", point)
plt.show()
