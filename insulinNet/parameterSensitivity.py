from copy import deepcopy

import insulinNet
import insulinParams
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


if __name__ == '__main__':
    print("Hello world")

    [speciesNames, tau, ymax, y0, w, n, EC50] = insulinParams.loadParams(diabetic=False)
    w[0] = 1  # turn insulin on
    tspan = [0, 10]
    
    sol = solve_ivp(insulinNet.ODEfunc, tspan, y0, args=(tau, ymax, w, n, EC50,))
    glu = sol.y[6].T
    glu_steady = glu[-1]
    
    # increase tau by 50%
    tau_inc = []
    for i in range(len(tau)):
        tau2 = deepcopy(tau)
        tau2[i] = 1.5 * tau[i]

        sol = solve_ivp(insulinNet.ODEfunc, tspan, y0, args=(tau2, ymax, w, n, EC50,))
        glu = sol.y[6].T
        glu2 = glu[-1]
        tau_inc.append(glu2 - glu_steady)
    
    # Decrease by 50%
    tau_dec = []
    for i in range(len(tau)):
        tau2 = deepcopy(tau)
        tau2[i] = 0.5 * tau[i]

        sol = solve_ivp(insulinNet.ODEfunc, tspan, y0, args=(tau2, ymax, w, n, EC50,))
        glu = sol.y[6].T
        glu2 = glu[-1]
        tau_dec.append(glu2 - glu_steady)

    # Increase by 50%
    ymax_inc = []
    for i in range(len(ymax)):
        ymax2 = deepcopy(ymax)
        ymax2[i] = 1.5 * ymax[i]

        sol = solve_ivp(insulinNet.ODEfunc, tspan, y0, args=(tau, ymax2, w, n, EC50,))
        glu = sol.y[6].T
        glu2 = glu[-1]
        ymax_inc.append(glu2 - glu_steady)

    # Decrease by 50%
    ymax_dec = []
    for i in range(len(ymax)):
        ymax2 = deepcopy(ymax)
        ymax2[i] = 0.5 * ymax[i]

        sol = solve_ivp(insulinNet.ODEfunc, tspan, y0, args=(tau, ymax2, w, n, EC50,))
        glu = sol.y[6].T
        glu2 = glu[-1]
        ymax_dec.append(glu2 - glu_steady)

    # Make 0.1
    y0_inc = []
    for i in range(len(y0)):
        y02 = deepcopy(y0)
        y02[i] = 0.1

        sol = solve_ivp(insulinNet.ODEfunc, tspan, y02, args=(tau, ymax, w, n, EC50,))
        glu = sol.y[6].T
        glu2 = glu[-1]
        y0_inc.append(glu2 - glu_steady)

    # Increase by 50%
    w_inc = []
    for i in range(len(w)):
        w2 = deepcopy(w)
        w2[i] = 1.5 * w[i]

        sol = solve_ivp(insulinNet.ODEfunc, tspan, y0, args=(tau, ymax, w2, n, EC50,))
        glu = sol.y[6].T
        glu2 = glu[-1]
        w_inc.append(glu2 - glu_steady)

    # Decrease by 50%
    w_dec = []
    for i in range(len(w)):
        w2 = deepcopy(w)
        w2[i] = 0.5 * w[i]

        sol = solve_ivp(insulinNet.ODEfunc, tspan, y0, args=(tau, ymax, w2, n, EC50,))
        glu = sol.y[6].T
        glu2 = glu[-1]
        w_dec.append(glu2 - glu_steady)

    # Increase by 0.3
    n_inc = []
    for i in range(len(n)):
        n2 = deepcopy(n)
        n2[i] = 0.3 + n[i]

        sol = solve_ivp(insulinNet.ODEfunc, tspan, y0, args=(tau, ymax, w, n2, EC50,))
        glu = sol.y[6].T
        glu2 = glu[-1]
        n_inc.append(glu2 - glu_steady)

    # Decrease by 0.3
    n_dec = []
    for i in range(len(n)):
        n2 = deepcopy(n)
        n2[i] = -0.3 + n[i]

        sol = solve_ivp(insulinNet.ODEfunc, tspan, y0, args=(tau, ymax, w, n2, EC50,))
        glu = sol.y[6].T
        glu2 = glu[-1]
        n_dec.append(glu2 - glu_steady)

    # Increase by 50%
    EC50_inc = []
    for i in range(len(EC50)):
        EC502 = deepcopy(EC50)
        EC502[i] = 1.5 * EC50[i]

        sol = solve_ivp(insulinNet.ODEfunc, tspan, y0, args=(tau, ymax, w, n, EC502,))
        glu = sol.y[6].T
        glu2 = glu[-1]
        EC50_inc.append(glu2 - glu_steady)

    # Decrease by 50%
    EC50_dec = []
    for i in range(len(EC50)):
        EC502 = deepcopy(EC50)
        EC502[i] = 0.5 * EC50[i]

        sol = solve_ivp(insulinNet.ODEfunc, tspan, y0, args=(tau, ymax, w, n, EC502,))
        glu = sol.y[6].T
        glu2 = glu[-1]
        EC50_dec.append(glu2 - glu_steady)

    
    fig, axis = plt.subplots(3, 2, figsize=(15,10))
    axis[0, 0].bar(speciesNames, tau_inc)
    axis[0, 0].set_title("tau Increase by 50%")

    axis[0, 1].bar(speciesNames, tau_dec)
    axis[0, 1].set_title("tau Decrease by 50%")

    axis[1, 0].bar(speciesNames, ymax_inc)
    axis[1, 0].set_title("ymax Increase by 50%")

    axis[1, 1].bar(speciesNames, ymax_dec)
    axis[1, 1].set_title("ymax Decrease by 50%")

    axis[2, 0].bar(speciesNames, y0_inc)
    axis[2, 0].set_title("y0 set to 0.1")

    fig.tight_layout()
    plt.show()

    reaction_names = ["Ins", "Ins->IR", "IR->IRS1", "mTORC1->IRS1", "IRS1->PKB",
                      "IR->PKB", "PKB->mTORC1", "PKB->GLUT4", "GLUT4->glucose"]

    fig, axis = plt.subplots(3, 2, figsize=(15,15))
    axis[0, 0].bar(reaction_names, w_inc)
    axis[0, 0].set_xticklabels(reaction_names, rotation=45)
    axis[0, 0].set_title("w Increase by 50%")

    axis[0, 1].bar(reaction_names, w_dec)
    axis[0, 1].set_xticklabels(reaction_names, rotation=45)
    axis[0, 1].set_title("w Decrease by 50%")

    axis[1, 0].bar(reaction_names, n_inc)
    axis[1, 0].set_xticklabels(reaction_names, rotation=45)
    axis[1, 0].set_title("n Increase by 0.3")

    axis[1, 1].bar(reaction_names, n_dec)
    axis[1, 1].set_xticklabels(reaction_names, rotation=45)
    axis[1, 1].set_title("n Decrease by 0.3")

    axis[2, 0].bar(reaction_names, EC50_inc)
    axis[2, 0].set_xticklabels(reaction_names, rotation=45)
    axis[2, 0].set_title("EC50 Increase by 50%")

    axis[2, 1].bar(reaction_names, EC50_dec)
    axis[2, 1].set_xticklabels(reaction_names, rotation=45)
    axis[2, 1].set_title("EC50 Decrease by 50%")

    fig.tight_layout()
    plt.show()

