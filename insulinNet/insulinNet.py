# exampleNet.py
# Automatically generated by Netflux on 26-Oct-2022
import numpy as np


def ODEfunc(t, y, tau, ymax, w, n, EC50):
    INS = 0
    IRprime= 1
    IR = 2
    IRS1= 3
    PKB = 4
    mTORC1 = 5
    GLUT4 = 6
    Glucose = 7

    dydt = np.zeros(8)

    dydt[INS] = (w[0] * ymax[INS] - y[INS]) / tau[INS]
    dydt[IRprime] = (act(y[INS], w[1], n[1], EC50[1]) * ymax[IRprime] - y[IRprime]) / tau[IRprime]
    dydt[IR] = (act(y[IRprime], w[2], n[2], EC50[2]) * ymax[IR] - y[IR]) / tau[IR]
    dydt[IRS1] = (AND(1, [act(y[IR], w[3], n[3], EC50[3]), act(y[IR], w[4], n[4], EC50[4])]) * ymax[IRS1] - y[IRS1]) / tau[IRS1]
    dydt[PKB] = (AND(1, [act(y[IRS1], w[5], n[5], EC50[5]), act(y[IR], w[6], n[6], EC50[6])]) * ymax[PKB] - y[PKB]) / tau[PKB]
    dydt[mTORC1] = (act(y[PKB], w[7], n[7], EC50[7]) * ymax[mTORC1] - y[mTORC1]) / tau[mTORC1]
    dydt[GLUT4] = (act(y[PKB], w[8], n[8], EC50[8]) * ymax[GLUT4] - y[GLUT4]) / tau[GLUT4]
    dydt[Glucose] = (act(y[GLUT4], w[9], n[9], EC50[9])) / tau[Glucose]
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