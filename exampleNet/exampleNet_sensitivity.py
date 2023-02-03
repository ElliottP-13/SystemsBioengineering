'''The following code is used to run a sensitivity analysis for the
hypertrophy network model'''

# %%
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import exampleNet
import exampleNet_params
from copy import deepcopy
import pandas as pd
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.cm as cm

[speciesNames, tau, ymax, y0, w, n, EC50] = exampleNet_params.loadParams()
N = len(speciesNames)

# %% Change input weights to 0.072 for sensitivity analysis
w = w.astype('float64')
w[:2] = 0.72

# %% Set all initial values to 0
y0[:] = 0

# %% Default Simulation
tspan = [0, 30]
sol = solve_ivp(exampleNet.ODEfunc, tspan, y0, args=(tau, ymax, w, n, EC50,))
t = sol.t
y = sol.y.T

# %% Plot the results to show steady state is reached for all species
fig, ax = plt.subplots()
ax.plot(t, y)
ax.set(xlabel='Time (h)', ylabel='Fractional activation')
ax.set_title("Baseline stimulation of Example Network Model")

# %% Perform interpolation using t and y: Example Case
# Note: This is important when attempting to add perturbations at specific time
# points
y_A = y[:, 0]

tnew = np.arange(0, tspan[1], 0.1)
y_A_interpolator = interp1d(t, y_A)
ynew = y_A_interpolator(tnew)  # use interpolation function returned by `interp1d`
plt.plot(t, y_A, 'o', tnew, ynew, '-')
plt.show()

# %% Perform interpolation for all species
tnew = np.arange(0, tspan[1], 0.1 / 6)
T = len(tnew)
y_interp = np.zeros([T, N])

for i in range(N):
    # Capture the values for each species in a new array
    y_i_interpolator = interp1d(t, y[:, i])
    y_interp[:, i] = y_i_interpolator(tnew)

# %% Capture the steady state value for each species
y_steady = y[-1, :]

# %% Establish array for perturbed simulation results
y_steady_kd = np.zeros([N, N])
sens = np.zeros([N, N])
delta_p = -1  # Parameter for knockdown

# %% for loop for iteration through knockdowns
for i in range(N):
    ymax_kd = deepcopy(ymax)
    ymax_kd[i] = (1 + delta_p) * ymax[i]

    sol_kd = solve_ivp(exampleNet.ODEfunc, tspan, y0, args=(tau, ymax_kd, w, n, EC50,))
    t = sol_kd.t
    y_kd = sol_kd.y.T
    y_steady_kd[:, i] = y_kd[-1, :]
    steady_diff = y_steady_kd[:, i] - y_steady
    sens[:, i] = steady_diff / (ymax_kd[i] - ymax[i]) * ymax[i] / np.transpose(ymax)

# %% Plot bar graph of E's sensitivity to knockouts
fig, ax = plt.subplots()
ax.bar(speciesNames, sens[-1, :])
ax.set(ylim=(-1, 1), xlabel='Knockout species', ylabel='Species E sensitivity to knockout species')

# %% Plot Sensitivity Matrix for all species

vcenter = 0
vmin, vmax = sens.min(), sens.max()
normalize = mcolors.TwoSlopeNorm(vcenter=vcenter, vmin=vmin, vmax=vmax)
colormap = cm.RdBu_r

sens_heatmap = pd.DataFrame(sens)
fig1, ax1 = plt.subplots()
fig1.set_size_inches(20.5, 14.5)
ax1 = sns.heatmap(sens_heatmap, norm=normalize, cmap=colormap, xticklabels=speciesNames, yticklabels=speciesNames)
ax1.set(title="Sensitivity Analsysis for Example Network Model")

plt.show()
# %%