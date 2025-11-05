import torch
import numpy as np
from util import plt

from scipy.optimize import curve_fit


plt.close("all")

plt.rcParams['figure.dpi'] = 250
plt.rcParams['savefig.dpi'] = 250
plt.rcParams['font.size'] = 18
plt.rc('legend', fontsize=15)
plt.rcParams['lines.linewidth'] = 3.5
msz = 14
handlelength = 3.0     # 2.75
borderpad = 0.25     # 0.15

linestyle_tuples = {
     'solid':                 '-',
     'dashdot':               '-.',
     
     'loosely dotted':        (0, (1, 10)),
     'dotted':                (0, (1, 1)),
     'densely dotted':        (0, (1, 1)),
     
     'long dash with offset': (5, (10, 3)),
     'loosely dashed':        (0, (5, 10)),
     'dashed':                (0, (5, 5)),
     'densely dashed':        (0, (5, 1)),

     'loosely dashdotted':    (0, (3, 10, 1, 10)),
     'dashdotted':            (0, (3, 5, 1, 5)),
     'densely dashdotted':    (0, (3, 1, 1, 1)),

     'dashdotdotted':         (0, (3, 5, 1, 5, 1, 5)),
     'loosely dashdotdotted': (0, (3, 10, 1, 10, 1, 10)),
     'densely dashdotdotted': (0, (3, 1, 1, 1, 1, 1))}

marker_list = ['o', 'd', 's', 'v', 'X', "*", "P", "^"]
style_list = ['-.', linestyle_tuples['dotted'], linestyle_tuples['densely dashdotted'],
              linestyle_tuples['densely dashed'], linestyle_tuples['densely dashdotdotted']]

def get_stats(ar):
    out = np.zeros((*ar.shape[-(ar.ndim - 1):], 2))
    out[..., 0] = np.mean(ar, axis=0)
    out[..., 1] = np.std(ar, axis=0)
    return out

# USER INPUT
FLAG_save_plots = True
FLAG_WIDE = not True
n_std = 2
plot_tol = 1e-7
SHIFT = 0
num_losses = 3

N_list = [256, 1024, 4096]
Noise_list = [0, 1, 3, 5, 10, 15, 20, 30]
Seed_list = [0, 1, 2, 3, 4]
exp_date = "2025-11-05"
load_prefix = "paper_sweep_three_phase"
plot_folder_base = "./results/" + exp_date + "/" + load_prefix


# Legend
legs = [r"$N=256$", r"$N=1024$", r"$N=4096$"]
legs_alt = ["Noisy Test", "Clean Test"]

# Colors
color_list = ['k', 'C3', 'C5', 'C1', 'C2', 'C0', 'C4', 'C6', 'C7', 'C8', 'C9'] # black, red, brown, orange, green, blue, purple, pink, gray, olive, cyan
    
if FLAG_WIDE:
    plt.rcParams['figure.figsize'] = [6.0, 4.0]     # [6.0, 4.0]
else:
    plt.rcParams['figure.figsize'] = [6.0, 6.0]     # [6.0, 4.0]


# Load data
plot_errors_raw = np.zeros((len(Seed_list), len(Noise_list), len(N_list), 2, num_losses)) # 2 for noisy and clean
for i, N in enumerate(N_list):
    for j, Noise in enumerate(Noise_list):
        for k, Seed in enumerate(Seed_list):
            plot_folder = plot_folder_base + "_N" + str(N) + "_Noise" + str(Noise) + "_Seed" + str(Seed) + "/"

            # Load
            plot_errors_raw[k,j,i,0,...] = torch.load(plot_folder + 'errors_test.pt', weights_only=True).numpy()
            plot_errors_raw[k,j,i,1,...] = torch.load(plot_folder + 'errors_test_clean.pt', weights_only=True).numpy()

# [Noise, N_train, CleanFlag, MeanOrStdev]
plot_errors = get_stats(plot_errors_raw[..., 0]) # L^1 loss only!
noise_plot = np.asarray(Noise_list) / 100.0

##################################
# Robustness Plots: Err vs noise, varying sample size on linear linear scale
##################################
def make_noise_sweep_plot(my_errors, fig_num=0, my_str="noisy"):
    """
    my_errors: (Noise, N_train, MeanOrStdev) array
    """
    plt.figure(fig_num)
    
    for i in range(len(N_list)):
        x = my_errors[:,i,0]
        twosigma = n_std*my_errors[:,i,1]
        lb = np.maximum(x - twosigma, plot_tol)
        ub = x + twosigma
    
        plt.plot(noise_plot, x, ls=style_list[i], color=color_list[i], marker=marker_list[i], markersize=msz, label=legs[i])
        plt.fill_between(noise_plot, lb, ub, facecolor=color_list[i], alpha=0.125)
    
    plt.ylim(0.26, 0.4)
    plt.xlabel(r'Noise Level')
    plt.ylabel(r'Average Relative $L^1$ Test Error')
    plt.legend(framealpha=1, loc='best', borderpad=borderpad,handlelength=handlelength).set_draggable(True)
    plt.grid(True, which="both")
    plt.tight_layout()
    if FLAG_save_plots:
        if FLAG_WIDE:
            plt.savefig("./results/" + exp_date + "/" + 'noise_sweep_wide_three_phase_' + my_str + '.pdf', format='pdf')
        else:
            plt.savefig("./results/" + exp_date + "/" + 'noise_sweep_three_phase_' + my_str + '.pdf', format='pdf')
    plt.show()

make_noise_sweep_plot(plot_errors[...,0,:], 0, "noisy")
make_noise_sweep_plot(plot_errors[...,1,:], 1, "clean")


##################################
# Stability Plots
##################################
stop = None if SHIFT == 0 else -SHIFT
y = plot_errors[1:stop,...] # remove zero
s = noise_plot[1:stop]

def model_power(s, E0, c, rho):
    """
    Model: offset power law E = E0 + c * sigma**rho
    """
    return E0 + c * np.power(np.maximum(s, 1e-15), rho)

def fit_power(sigma, err):
    """Initial guesses: E0≈min(err), rho≈1, c based on first step"""
    p0 = (float(err.min()), float((err.max()-err.min())/(sigma.max()**1 if sigma.max()>0 else 1)), 1.0)
    bounds = (0.0, [np.inf, np.inf, 3.0])  # rho capped to something reasonable
    E0, c, rho = curve_fit(model_power, sigma, err, p0=p0, bounds=bounds, maxfev=10000)[0]
    return dict(E0=E0, c=c, rho=rho)

def model_log(s, E0, c, rho):
    """E(s) = E0 + c * (log(1/s))^{-rho}"""
    s = np.asarray(s, dtype=float)
    return E0 + c * np.power(np.log(1 / s), -rho)

def linfit(x, y):
    A = np.vstack([np.ones_like(x), x]).T
    b0, b1 = np.linalg.lstsq(A, y, rcond=None)[0]
    yhat = b0 + b1*x
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res/ss_tot
    return float(b0), float(b1), float(r2)

def fit_log(s, y):
    x = np.log(np.log(1/s))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    E0_max = ymin - 1e-10
    E0_min = ymin - 0.25*(ymax - ymin) - 1e-10
    best = None
    for E0 in np.linspace(E0_min, E0_max, 400):
        y_shift = y - E0
        if np.any(y_shift <= 0): continue
        z = np.log(y_shift)
        b0, slope, r2 = linfit(x, z)  # slope ~ -rho
        c = float(np.exp(b0))
        rho = -slope
        yhat = E0 + c * np.power(np.log(1/s), -rho)
        loss = float(np.mean((y - yhat)**2))
        if (best is None) or (loss < best['loss']):
            best = dict(E0=float(E0), c=c, rho=rho, slope=float(slope), R2=float(r2), loss=loss)
    return dict(E0=best['E0'], c=best['c'], rho=best['rho'])

param_power = [fit_power(s, y[:, j, 0,0]) for j in range(y.shape[1])]
param_log = [fit_log(s, y[:, j,0,0]) for j in range(y.shape[1])]

param_power_clean = [fit_power(s, y[:, j, 1,0]) for j in range(y.shape[1])]
param_log_clean = [fit_log(s, y[:, j,1,0]) for j in range(y.shape[1])]

for d, fp in zip([param_power,param_power_clean,param_log,param_log_clean],
                 ["Power Noisy","Power Clean", "Log Noisy", "Log Clean"]):
    print(fp)
    for i, fit in enumerate(d, start=1):
        print(f"Curve {i}: E0 = {fit['E0']:.4f}, c = {fit['c']:.4f}, rho = {fit['rho']:.3f}")


def make_noise_fit_power(my_errors, x, d, model, fig_num=0, my_str="noisy"):
    """
    my_errors: (Noise, N_train, MeanOrStdev) array
    """
    plt.figure(fig_num)
    for i in range(len(N_list)):
        plt.loglog(x, model(x, **d[i]) - d[i]['E0'], ls='-', color='purple') # darkgray

        y = my_errors[:,i,0] - d[i]['E0']
        twosigma = n_std*my_errors[:,i,1]
        lb = np.maximum(y - twosigma, plot_tol)
        ub = y + twosigma
    
        plt.loglog(x, y, ls=style_list[i], color=color_list[i], marker=marker_list[i], markersize=msz, label=legs[i])
        plt.fill_between(x, lb, ub, facecolor=color_list[i], alpha=0.125)
    
    # plt.ylim(1e-4, 2e-1)
    plt.xlabel(r'$\delta$')
    plt.ylabel(r'$\mathrm{Err}_{\delta,N} - \mathrm{Err}_{0,N}$')
    plt.legend(framealpha=1, loc='best', borderpad=borderpad,handlelength=handlelength).set_draggable(True)
    plt.grid(True, which="both")
    plt.tight_layout()
    if FLAG_save_plots:
        if FLAG_WIDE:
            plt.savefig("./results/" + exp_date + "/" + 'noise_power_wide_three_phase_' + my_str + '.pdf', format='pdf')
        else:
            plt.savefig("./results/" + exp_date + "/" + 'noise_power_three_phase_' + my_str + '.pdf', format='pdf')
    plt.show()
    
make_noise_fit_power(y[...,0,:], s, param_power, model_power, 10, "noisy")
make_noise_fit_power(y[...,1,:], s, param_power_clean, model_power, 11, "clean")


def make_noise_fit_log(my_errors, x, d, model, fig_num=0, my_str="noisy"):
    """
    my_errors: (Noise, N_train, MeanOrStdev) array
    """
    xplot = -np.log2(x)
    
    plt.figure(fig_num)
    for i in range(len(N_list)):
        plt.loglog(xplot, model(x, **d[i]) - d[i]['E0'], ls='-', color='purple')

        y = my_errors[:,i,0] - d[i]['E0']
        twosigma = n_std*my_errors[:,i,1]
        lb = np.maximum(y - twosigma, plot_tol)
        ub = y + twosigma
    
        plt.loglog(xplot, y, ls=style_list[i], color=color_list[i], marker=marker_list[i], markersize=msz, label=legs[i])
        plt.fill_between(xplot, lb, ub, facecolor=color_list[i], alpha=0.125)
    
    plt.ylim(1e-3, 1e-1)
    plt.xlabel(r'$\log(1/\delta)$')
    plt.ylabel(r'$\mathrm{Err}_{\delta,N} - \mathrm{Err}_{0,N}$')
    plt.legend(framealpha=1, loc='best', borderpad=borderpad,handlelength=handlelength).set_draggable(True)
    plt.grid(True, which="both")
    plt.tight_layout()
    if FLAG_save_plots:
        if FLAG_WIDE:
            plt.savefig("./results/" + exp_date + "/" + 'noise_log_wide_three_phase_' + my_str + '.pdf', format='pdf')
        else:
            plt.savefig("./results/" + exp_date + "/" + 'noise_log_three_phase_' + my_str + '.pdf', format='pdf')
    plt.show()

make_noise_fit_log(y[...,0,:], s, param_log, model_log, 20, "noisy")
make_noise_fit_log(y[...,1,:], s, param_log_clean, model_log, 21, "clean")
