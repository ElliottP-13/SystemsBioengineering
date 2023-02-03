# exampleNet.py
# Automatically generated by Netflux on 26-Oct-2022
import numpy as np


def ODEfunc(t, y, tau, ymax, w, n, EC50):
    A = 0
    B = 1
    C = 2
    D = 3
    E = 4
    dydt = np.zeros(5)
    dydt[A] = (w[0] * ymax[A] - y[A]) / tau[A]
    dydt[B] = (w[1] * ymax[B] - y[B]) / tau[B]
    dydt[C] = (OR(act(y[A], w[2], n[2], EC50[2]), act(y[E], w[5], n[5], EC50[5])) * ymax[C] - y[C]) / tau[C]
    dydt[D] = (act(y[B], w[3], n[3], EC50[3]) * ymax[D] - y[D]) / tau[D]
    dydt[E] = (AND(w[4], [act(y[C], w[4], n[4], EC50[4]), inhib(y[D], w[4], n[4], EC50[4])]) * ymax[E] - y[E]) / tau[E]
    return dydt


# utility functions

def act(x, w, n, EC50):
    # hill activation function with parameters w (weight), n (Hill coeff), EC50
    beta = ((EC50 ** n) - 1) / (2 * EC50 ** n - 1)
    K = (beta - 1) ** (1 / n)
    fact = w * (beta * x ** n) / (K ** n + x ** n)
    if fact > w:
        fact = w
    return fact


def inhib(x, w, n, EC50):
    # inverse hill function with parameters w (weight), n (Hill coeff), EC50
    finhib = w - act(x, w, n, EC50)
    return finhib


def OR(x, y):
    # OR logic gate
    z = x + y - x * y
    return z


def AND(w, reactList):
    # AND logic gate, multiplying all of the reactants together
    if w == 0:
        z = 0
    else:
        p = np.array(reactList).prod()
        z = p / w ** (len(reactList) - 2)
    return z
