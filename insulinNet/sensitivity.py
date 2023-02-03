import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import insulinNet
import insulinParams
from copy import deepcopy
import pandas as pd
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.cm as cm


if __name__ == "__main__":
    print("hello world")

    [speciesNames, tau, ymax, y0, w, n, EC50] = insulinParams.loadParams(diabetic=False)

    tspan = [0, 10]

    # run default
    w[0] = 1  # turn insulin on
    sol = solve_ivp(insulinNet.ODEfunc, tspan, y0, args=(tau, ymax, w, n, EC50,))
    y = sol.y.T
    y_steady = y[-1, :]

    N = len(speciesNames)
    y_steady_kd = np.zeros([N, N])
    sens = np.zeros([N, N])
    delta_p = -1  # Parameter for knockdown

    # Run knockouts without specific enzyme
    for i in range(N):
        ymax_kd = deepcopy(ymax)
        ymax_kd[i] = (1 + delta_p) * ymax[i]

        sol_kd = solve_ivp(insulinNet.ODEfunc, tspan, y0, args=(tau, ymax_kd, w, n, EC50,))
        t = sol_kd.t
        y_kd = sol_kd.y.T
        y_steady_kd[:, i] = y_kd[-1, :]
        steady_diff = y_steady_kd[:, i] - y_steady
        sens[:, i] = steady_diff / (ymax_kd[i] - ymax[i]) * ymax[i] / np.transpose(ymax)

    # %% Plot bar graph of E's sensitivity to knockouts
    fig, ax = plt.subplots()
    ax.bar(speciesNames, sens[-1, :])
    ax.set(ylim=(-1, 1), xlabel='Knockout species', ylabel='Glucose sensitivity to knockout species')

    # %% Plot Sensitivity Matrix for all species

    vcenter = 0
    vmin, vmax = sens.min(), sens.max()
    print(vcenter, vmin, vmax)
    normalize = mcolors.TwoSlopeNorm(vcenter=vcenter, vmin=vmin, vmax=vmax)
    colormap = cm.RdBu_r

    sens_heatmap = pd.DataFrame(sens)
    fig1, ax1 = plt.subplots()
    ax1 = sns.heatmap(sens_heatmap, norm=normalize, cmap=colormap, xticklabels=speciesNames, yticklabels=speciesNames)
    ax1.set(title="Sensitivity Analsysis for Insulin Network Model")

    plt.show()
    # %%

