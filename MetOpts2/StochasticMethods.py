import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["figure.figsize"] = (20,10)

def f(a, b, x, y, n):
    sum = 0
    for i in range(n):
        sum += (a * x[i] + b - y[i]) ** 2
    return sum / n

def grad(a, b, x, y, batchSize, n):
    sum_a = 0
    sum_b = 0
    for i in range(batchSize):
        temp = a * x[i] + b - y[i]
        sum_a += 2 * x[i] * temp
        sum_b += 2 * temp
    res = np.array([sum_a / n, sum_b / n])
    return res

def SGD(x, y, n, batchSize, learningRate, epochs, startValue):
    rng = np.random.default_rng()
    xSize = x.size
    xy = np.c_[x.reshape(n, -1), y.reshape(n, 1)]
    point_a = startValue[0]
    point_b = startValue[1]
    points = [[point_a, point_b]]
    for i in range(epochs):
        rng.shuffle(xy)
        xBatch = xy[0:batchSize, :-1].flatten()
        yBatch = xy[0:batchSize, -1:].flatten()
        gradient = grad(point_a, point_b, xBatch, yBatch, batchSize, n)
        point_a -= learningRate * gradient[0]
        point_b -= learningRate * gradient[1]
        points.append([point_a, point_b])
    return [point_a, point_b, points]

def SgdMomentum(x, y, n, batchSize, learningRate, epochs, startValue):
    rng = np.random.default_rng()
    xSize = x.size
    xy = np.c_[x.reshape(n, -1), y.reshape(n, 1)]
    point_a = startValue[0]
    point_b = startValue[1]
    points = [[point_a, point_b]]
    i = 1
    B = 0.7
    xBatch = xy[0:batchSize, :-1].flatten()
    yBatch = xy[0:batchSize, -1:].flatten()
    gradient = 0
    while(True):
        rng.shuffle(xy)
        xBatch = xy[0:batchSize, :-1].flatten()
        yBatch = xy[0:batchSize, -1:].flatten()
        if (i == 1): gradient = grad(point_a, point_b, xBatch, yBatch, batchSize, n)
        else: gradient = B * gradient + ((1 - B) * grad(point_a, point_b, xBatch, yBatch, batchSize, n))
        point_a -= learningRate * gradient[0]
        point_b -= learningRate * gradient[1]
        points.append([point_a, point_b])
        if abs(f(points[i - 1][0], points[i - 1][1], x, y, n) - f(point_a, point_b, x, y, n)) < 0.00001:
            break
        i = i + 1
    return [point_a, point_b, points]

def SgdNesterovMomentum(x, y, n, batchSize, learningRate, epochs, startValue):
    rng = np.random.default_rng()
    xSize = x.size
    xy = np.c_[x.reshape(n, -1), y.reshape(n, 1)]
    point_a = startValue[0]
    point_b = startValue[1]
    points = [[point_a, point_b]]
    i = 1
    B = 0.3
    alpha = 0.7

    gradient = 0
    while(True):
        rng.shuffle(xy)
        xBatch = xy[0:batchSize, :-1].flatten()
        yBatch = xy[0:batchSize, -1:].flatten()
        if (i == 1): gradient = grad(point_a, point_b, xBatch, yBatch, batchSize, n)
        else: gradient = B * gradient + alpha * (grad(point_a - B * gradient[0], point_b - B * gradient[1],
                                                      xBatch, yBatch, batchSize, n))
        point_a -= learningRate * gradient[0]
        point_b -= learningRate * gradient[1]
        points.append([point_a, point_b])
        if abs(f(points[i - 1][0], points[i - 1][1], x, y, n) - f(point_a, point_b, x, y, n)) < 0.00001:
            break
        i = i + 1
    return [point_a, point_b, points]

def SgdRmsProp(x, y, n, batchSize, learningRate, epochs, startValue):
    rng = np.random.default_rng()
    xSize = x.size
    xy = np.c_[x.reshape(n, -1), y.reshape(n, 1)]
    point_a = startValue[0]
    point_b = startValue[1]
    points = [[point_a, point_b]]
    i = 1
    B = 0.7
    xBatch = xy[0:batchSize, :-1].flatten()
    yBatch = xy[0:batchSize, -1:].flatten()
    while(True):
        rng.shuffle(xy)
        xBatch = xy[0:batchSize, :-1].flatten()
        yBatch = xy[0:batchSize, -1:].flatten()
        if (i == 1): gradient = -1 * grad(point_a, point_b, xBatch, yBatch, batchSize, n)
        else: gradient = B * gradient + ((1 - B) * (np.array(grad(point_a, point_b, xBatch, yBatch, batchSize, n)) ** 2))
        point_a -= (learningRate / math.sqrt(gradient[0])) * grad(point_a, point_b, xBatch, yBatch, batchSize, n)[0]
        point_b -= (learningRate / math.sqrt(gradient[1])) * grad(point_a, point_b, xBatch, yBatch, batchSize, n)[1]
        points.append([point_a, point_b])
        if abs(f(points[i - 1][0], points[i - 1][1], x, y, n) - f(point_a, point_b, x, y, n)) < 0.0001:
            break
        i = i + 1
    return [point_a, point_b, points]
n = 100
a = 2
b = 6

rng = np.random.RandomState(1)
x = 5 * rng.rand(n, 1)
y = b + a * x + np.random.randn(n , 1)

result = SgdRmsProp(x, y, n, 10, 0.2, 10000, [1, 4])
points = np.array(result[2])
ax = plt.figure().add_subplot()
lineX = np.arange(x.min(), x.max(), 0.1)
lineY = result[0] * lineX + result[1]
ax.plot(f(points[:, 0], points[:, 1], x, y, n))
print("Minimum: a = ", result[0], " b = ", result[1])
plt.figure()
plt.scatter(x, y)
plt.plot(lineX, lineY)
plt.show()