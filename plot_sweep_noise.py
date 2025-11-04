import torch
import numpy as np
from util import plt

from numpy.polynomial.polynomial import polyfit


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
FLAG_save_plots = not True
FLAG_WIDE = not True
n_std = 2
plot_tol = 1e-7
SHIFT = 2
num_losses = 3

N_list = [256, 1024, 4096]
Noise_list = [0, 1, 3, 5, 10, 15, 20, 30]
Seed_list = [0, 1, 2, 3, 4]
exp_date = "2025-10-23"
load_prefix = "paper_sweep"
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


# Plot: Err vs noise, varying sample size on linear linear scale
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
    
    # plt.xlim(left=9e0)
    plt.ylim(0.26, 0.57)
    plt.xlabel(r'Noise Level')
    plt.ylabel(r'Average Relative $L^1$ Test Error')
    plt.legend(framealpha=1, loc='best', borderpad=borderpad,handlelength=handlelength).set_draggable(True)
    plt.grid(True, which="both")
    plt.tight_layout()
    if FLAG_save_plots:
        if FLAG_WIDE:
            plt.savefig("./results/" + exp_date + "/" + 'noise_sweep_wide_' + my_str + '.pdf', format='pdf')
        else:
            plt.savefig("./results/" + exp_date + "/" + 'noise_sweep_' + my_str + '.pdf', format='pdf')
    plt.show()

make_noise_sweep_plot(plot_errors[...,0,:], 0, "noisy")
make_noise_sweep_plot(plot_errors[...,1,:], 1, "clean")


# Log plots for Holder stability, remove zero noise
zero_noise_errors = plot_errors_raw[:,0,...,0]
dif_errors = get_stats(plot_errors_raw[:,1:, ..., 0] - zero_noise_errors[:,None,...]) # L^1 loss only!
noise_plot = np.asarray(Noise_list[1:]) / 100.0

# Least square fit to error data
nplot = noise_plot[SHIFT:]
nplota = noise_plot

def get_slopes(array_2d, my_str="noisy", nplot=nplot, nplota=nplota, exp_date=exp_date, SHIFT=SHIFT):
    linefit = polyfit(np.log2(nplot), np.log2(array_2d[SHIFT:,...]), 1)
    lineplota = linefit[0,...] + linefit[1,...]*np.log2(nplota)[:,None]
    my_slopes = -linefit[-1]
    print("Least square slope fit is (" + my_str + "): ")
    print(my_slopes)
    # TODO
    # np.save("./results/" + exp_date + "/" + 'rate_ls_noise_sweep_' + my_str + '.npy', linefit)
    return my_slopes, linefit, lineplota

my_noisy_slopes = get_slopes(dif_errors[...,0,0], "noisy")
my_clean_slopes = get_slopes(dif_errors[...,1,0], "clean")

my_noisy_slopes_log = get_slopes(dif_errors[...,0,0], "noisy", nplot=-np.log2(nplot), nplota=-np.log2(nplota))
my_clean_slopes_log = get_slopes(dif_errors[...,1,0], "clean", nplot=-np.log2(nplot), nplota=-np.log2(nplota))


def make_noise_sweep_plot_log(my_errors, my_slopes, fig_num=0, my_str="noisy"):
    """
    my_errors: (Noise, N_train, MeanOrStdev) array
    """
    plt.figure(fig_num)
    for i in range(len(N_list)):
        # ref = noise_levels**0.5 * errors[0]/(noise_levels[0]**0.5)

        plt.loglog(-np.log2(nplota), 2**my_slopes[-1][...,i], ls='-', color='darkgray')

        x = my_errors[:,i,0]
        twosigma = n_std*my_errors[:,i,1]
        lb = np.maximum(x - twosigma, plot_tol)
        ub = x + twosigma
    
        plt.loglog(-np.log2(noise_plot), x, ls=style_list[i], color=color_list[i], marker=marker_list[i], markersize=msz, label=legs[i])
        # plt.fill_between(-np.log2(noise_plot), lb, ub, facecolor=color_list[i], alpha=0.125)
    
    # plt.xlim(left=9e0)
    # plt.ylim(0.26, 0.57)
    plt.xlabel(r'$\log(1/\delta)$')
    plt.ylabel(r'Test Error Shifted By Noiseless Error')
    plt.legend(framealpha=1, loc='best', borderpad=borderpad,handlelength=handlelength).set_draggable(True)
    plt.grid(True, which="both")
    plt.tight_layout()
    if FLAG_save_plots:
        if FLAG_WIDE:
            plt.savefig("./results/" + exp_date + "/" + 'noise_sweep_wide_' + my_str + '.pdf', format='pdf')
        else:
            plt.savefig("./results/" + exp_date + "/" + 'noise_sweep_' + my_str + '.pdf', format='pdf')
    plt.show()
    
make_noise_sweep_plot_log(dif_errors[...,0,:], my_noisy_slopes_log, "noisy")
make_noise_sweep_plot_log(dif_errors[...,1,:], my_clean_slopes_log, "clean")


# =============================================================================
# # Plot: Err vs Sample size, varying clean/noisy test for fixed noise
# def make_data_sweep_plot_fixed_noise(my_errors, noise_idx, noise_val):
#     """
#     my_errors: (N_train, CleanOrNot, MeanOrStdev) array
#     """
#     
#     plt.figure(noise_idx)
#     
#     for i in range(2):
#         x = my_errors[:,i,0]
#         twosigma = n_std*my_errors[:,i,1]
#         lb = np.maximum(x - twosigma, plot_tol)
#         ub = x + twosigma
#     
#         plt.loglog(N_list, x, ls=style_list[i], color=color_list[i], marker=marker_list[i], markersize=msz, label=legs_alt[i])
#         plt.fill_between(N_list, lb, ub, facecolor=color_list[i], alpha=0.125)
#     
#     plt.xlim(left=9e0)
#     plt.ylim(top=1e0)
#     plt.xlabel(r'Sample Size')
#     plt.ylabel(r'Average Relative $L^1$ Test Error')
#     plt.legend(framealpha=1, loc='best', borderpad=borderpad,handlelength=handlelength).set_draggable(True)
#     plt.grid(True, which="both")
#     plt.tight_layout()
#     if FLAG_save_plots:
#         if FLAG_WIDE:
#             plt.savefig("./results/" + exp_date + "/" + 'data_sweep_wide_noise' + str(noise_val) + '.pdf', format='pdf')
#         else:
#             plt.savefig("./results/" + exp_date + "/" + 'data_sweep_noise' + str(noise_val) + '.pdf', format='pdf')
#     plt.show()
# 
# for i in range(len(Noise_list)):
#     make_data_sweep_plot_fixed_noise(plot_errors[:, i, ... ], i, Noise_list[i])
# 
# =============================================================================
