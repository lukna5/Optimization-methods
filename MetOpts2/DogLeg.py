import math
import numpy.linalg as ln
import numpy as np
from matplotlib import pyplot as plt


def get_m(fk, p, g, B):
    return fk + np.dot(p.transpose(), g) + 0.5 * np.dot(np.dot(p.transpose(), B), p)


def grad(a, b, c):
    return np.dot(jac(a, b, c).transpose(), residuals(a, b, c))


def residuals(a, b, c):
    res = np.zeros(n)
    for i in range(n):
        res[i] = (fx(a, b, c, tochki[i][0]) - tochki[i][1])
    return res


def f_squares(a, b, c):
    sum = 0
    for i in range(n):
        sum += (fx(a, b, c, tochki[i][0]) - tochki[i][1]) ** 2
    return sum


def f_squares_x(x):
    a = x[0]
    b = x[1]
    c = x[2]
    sum = 0
    for i in range(n):
        sum += (fx(a, b, c, tochki[i][0]) - tochki[i][1]) ** 2
    return sum


def fx(a, b, c, x):
    return a * x ** 2 + b * x + c


def real_fx(x):
    return 6 * x ** 2 + 3 * x + 5


def jac(a, b, c):
    h = 0.0001
    jac1 = np.random.random((n, 3))
    for i in range(n):
        jac1[i][0] = (fx(a, b, c, tochki[i][0]) - fx(a + h, b, c, tochki[i][0])) / h
        jac1[i][1] = (fx(a, b, c, tochki[i][0]) - fx(a, b + h, c, tochki[i][0])) / h
        jac1[i][2] = (fx(a, b, c, tochki[i][0]) - fx(a, b, c + h, tochki[i][0])) / h
    return -1 * jac1


def hess(a, b, c):
    jac1 = jac(a, b, c)
    return 2 * np.dot(jac1.transpose(), jac1)


def dogleg_method(Hk, gk, Bk, trust_radius):
    pB = -np.dot(Hk, gk)
    norm_pB = math.sqrt(np.dot(pB, pB))

    if norm_pB <= trust_radius:
        return pB

    pU = - (np.dot(gk, gk) / np.dot(gk, np.dot(Bk, gk))) * gk
    dot_pU = np.dot(pU, pU)
    norm_pU = math.sqrt(dot_pU)
    if norm_pU >= trust_radius:
        return trust_radius * pU / norm_pU

    pB_pU = pB - pU
    dot_pB_pU = np.dot(pB_pU, pB_pU)
    dot_pU_pB_pU = np.dot(pU, pB_pU)
    fact = dot_pU_pB_pU ** 2 - dot_pB_pU * (dot_pU - trust_radius ** 2)
    tau = (-dot_pU_pB_pU + math.sqrt(fact)) / dot_pB_pU
    return pU + tau * pB_pU


def trust_region_dogleg(x0, initial_trust_radius=1.0,
                        max_trust_radius=100.0, eta=0.15, gtol=1e-4,
                        maxiter=100):
    xk = x0
    trust_radius = initial_trust_radius
    k = 0
    while True:
        a = xk[0]
        b = xk[1]
        c = xk[2]
        gk = grad(a, b, c)
        Bk = hess(a, b, c)
        Hk = np.linalg.inv(Bk)
        pk = dogleg_method(Hk, grad(a, b, c), Bk, trust_radius)
        act_red = f_squares_x(xk) - f_squares_x(xk + pk)
        pred_red = get_m(f_squares(a, b, c), np.array([0, 0, 0]), gk, Bk) - get_m(f_squares(a, b, c), pk, gk, Bk)

        if pred_red == 0.0:
            act_pre = 1e99
        else:
            act_pre = act_red / pred_red

        norm_pk = math.sqrt(np.dot(pk, pk))

        if act_pre < 1 / 4:
            trust_radius = 0.25 * norm_pk
        else:
            if act_pre > 0.75 and norm_pk == trust_radius:
                trust_radius = min(2.0 * trust_radius, max_trust_radius)
            else:
                trust_radius = trust_radius

        if act_pre > eta:
            xk = xk + pk
        else:
            xk = xk

        if ln.norm(gk) < gtol:
            break

        if k >= maxiter:
            break
        k = k + 1
    return xk

n = 1000
a1 = 0.7
b1 = 3
c1 = 2
tochki = np.random.random((n, 2))
rng = np.random.RandomState(1)
x1 = 9 * rng.rand(n, 1).flatten()
y1 = np.array(a1 * x1 * x1 + b1 * x1 + c1 + np.array(np.random.randn(n , 1).flatten()))
for i in range(n):
    tochki[i][0] = x1[i]
    tochki[i][1] = y1[i]
print(trust_region_dogleg([1, 1, 1]))
fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(x1, y1)
ax.grid()