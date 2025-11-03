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
FLAG_save_plots = True
FLAG_WIDE = not True
n_std = 2
plot_tol = 1e-7
SHIFT = 2
num_losses = 3

N_list = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 9500]
Noise_list = [0, 3, 10, 30]
Seed_list = [0, 1, 2, 3, 4]
exp_date = "2025-10-23"
load_prefix = "paper_sweep"
plot_folder_base = "./results/" + exp_date + "/" + load_prefix


# Legend
legs = [r"$0\%$", r"$3\%$", r"$10\%$", r"$30\%$"]
legs_alt = ["Noisy Test", "Clean Test"]

# Colors
color_list = ['k', 'C3', 'C5', 'C1', 'C2', 'C0', 'C4', 'C6', 'C7', 'C8', 'C9'] # black, red, brown, orange, green, blue, purple, pink, gray, olive, cyan
    
if FLAG_WIDE:
    plt.rcParams['figure.figsize'] = [6.0, 4.0]     # [6.0, 4.0]
else:
    plt.rcParams['figure.figsize'] = [6.0, 6.0]     # [6.0, 4.0]


# Load data
plot_errors = np.zeros((len(Seed_list), len(N_list), len(Noise_list), 2, num_losses)) # 2 for noisy and clean
for i, N in enumerate(N_list):
    for j, Noise in enumerate(Noise_list):
        for k, Seed in enumerate(Seed_list):
            plot_folder = plot_folder_base + "_N" + str(N) + "_Noise" + str(Noise) + "_Seed" + str(Seed) + "/"

            # Load
            plot_errors[k,i,j,0,...] = torch.load(plot_folder + 'errors_test.pt', weights_only=True).numpy()
            plot_errors[k,i,j,1,...] = torch.load(plot_folder + 'errors_test_clean.pt', weights_only=True).numpy()

# [N_train, Noise, CleanFlag, MeanOrStdev]
plot_errors = get_stats(plot_errors[..., 0]) # L^1 loss only!


# Experimental rates of convergence table
eocBoch = np.zeros([len(N_list)-1, *plot_errors.shape[1:-1]])
for i in range(len(eocBoch)):
    eocBoch[i,...] = np.log2(plot_errors[i,...,0]/plot_errors[i + 1,...,0])/np.log2(N_list[i + 1]/N_list[i])
print("\nEOC is: ")
print(eocBoch)
np.save("./results/" + exp_date + "/" + "rate_table_L1_data_sweep.npy", eocBoch)


# Least square fit to error data
nplot = N_list[SHIFT:]
nplota = N_list

def get_slopes(array_2d, my_str="noisy", nplot=nplot, nplota=nplota, exp_date=exp_date, SHIFT=SHIFT):
    linefit = polyfit(np.log2(nplot), np.log2(array_2d[SHIFT:,...]), 1)
    lineplota = linefit[0,...] + linefit[1,...]*np.log2(nplota)[:,None]
    my_slopes = -linefit[-1]
    print("Least square slope fit is (" + my_str + "): ")
    print(my_slopes)
    np.save("./results/" + exp_date + "/" + 'rate_ls_data_sweep_' + my_str + '.npy', linefit)
    return my_slopes, linefit, lineplota

my_noisy_slopes = get_slopes(plot_errors[...,0,0], "noisy")
my_clean_slopes = get_slopes(plot_errors[...,1,0], "clean")


# Plot: Err vs Sample size, varying noise level
def make_data_sweep_plot(my_errors, fig_num=0, my_str="noisy"):
    """
    my_errors: (N_train, Noise, MeanOrStdev) array
    """
    
    plt.figure(fig_num)
    
    for i in range(len(Noise_list)):
        x = my_errors[:,i,0]
        twosigma = n_std*my_errors[:,i,1]
        lb = np.maximum(x - twosigma, plot_tol)
        ub = x + twosigma
    
        plt.loglog(N_list, x, ls=style_list[i], color=color_list[i], marker=marker_list[i], markersize=msz, label=legs[i])
        plt.fill_between(N_list, lb, ub, facecolor=color_list[i], alpha=0.125)
    
    plt.xlim(left=9e0)
    plt.ylim(top=1e0)
    plt.xlabel(r'Sample Size')
    plt.ylabel(r'Average Relative $L^1$ Test Error')
    plt.legend(framealpha=1, loc='best', borderpad=borderpad,handlelength=handlelength).set_draggable(True)
    plt.grid(True, which="both")
    plt.tight_layout()
    if FLAG_save_plots:
        if FLAG_WIDE:
            plt.savefig("./results/" + exp_date + "/" + 'data_sweep_wide_' + my_str + '.pdf', format='pdf')
        else:
            plt.savefig("./results/" + exp_date + "/" + 'data_sweep_' + my_str + '.pdf', format='pdf')
    plt.show()

make_data_sweep_plot(plot_errors[...,0,:], 0, "noisy")
make_data_sweep_plot(plot_errors[...,1,:], 1, "clean")


# Plot: Err vs Sample size, varying clean/noisy test for fixed noise
def make_data_sweep_plot_fixed_noise(my_errors, noise_idx, noise_val):
    """
    my_errors: (N_train, CleanOrNot, MeanOrStdev) array
    """
    
    plt.figure(noise_idx)
    
    for i in range(2):
        x = my_errors[:,i,0]
        twosigma = n_std*my_errors[:,i,1]
        lb = np.maximum(x - twosigma, plot_tol)
        ub = x + twosigma
    
        plt.loglog(N_list, x, ls=style_list[i], color=color_list[i], marker=marker_list[i], markersize=msz, label=legs_alt[i])
        plt.fill_between(N_list, lb, ub, facecolor=color_list[i], alpha=0.125)
    
    plt.xlim(left=9e0)
    plt.ylim(top=1e0)
    plt.xlabel(r'Sample Size')
    plt.ylabel(r'Average Relative $L^1$ Test Error')
    plt.legend(framealpha=1, loc='best', borderpad=borderpad,handlelength=handlelength).set_draggable(True)
    plt.grid(True, which="both")
    plt.tight_layout()
    if FLAG_save_plots:
        if FLAG_WIDE:
            plt.savefig("./results/" + exp_date + "/" + 'data_sweep_wide_noise' + str(noise_val) + '.pdf', format='pdf')
        else:
            plt.savefig("./results/" + exp_date + "/" + 'data_sweep_noise' + str(noise_val) + '.pdf', format='pdf')
    plt.show()

for i in range(len(Noise_list)):
    make_data_sweep_plot_fixed_noise(plot_errors[:, i, ... ], i, Noise_list[i])

# %% L^p loss plots for different p is not interesting
# # Plot: Err vs Sample size vs Loss
# my_plot = plot_errors[:, 1, 0, :,:]
# plt.figure(0)

# # round_slopes = np.round(my_slopes, 2)
# # plt.loglog(N_list, 0.9*2**lineplota[...,0], ls='--', color='darkgray', label=fr'$N^{{-{round_slopes[0]:.2f}}}$')
# # plt.loglog(N_list, 1.07*2**lineplota[...,1], ls='-', color='darkgray', label=fr'$N^{{-{round_slopes[1]:.2f}}}$')

# plt.loglog(N_list, 1.7*np.asarray(N_list)**(-0.3), ls='-', color='darkgray')

# for i in range(num_losses):
#     twosigma = n_std*my_plot[...,i,1]
#     lb = np.maximum(my_plot[...,i,0] - twosigma, plot_tol)
#     ub = my_plot[...,i,0] + twosigma

#     plt.loglog(N_list, my_plot[...,i,0], ls=style_list[i], color=color_list[i], marker=marker_list[i], markersize=msz, label=legs[i])
#     plt.fill_between(N_list, lb, ub, facecolor=color_list[i], alpha=0.125)

# plt.xlim(left=9e0)
# plt.xlabel(r'Sample Size')
# plt.ylabel(r'Average Relative $L^p$ Test Error')
# plt.legend(framealpha=1, loc='best', borderpad=borderpad,handlelength=handlelength).set_draggable(True)
# plt.grid(True, which="both")
# plt.tight_layout()
# if FLAG_save_plots:
#     if FLAG_WIDE:
#         plt.savefig("./results/" + exp_date + "/" + 'samplesize_wide' + '.pdf', format='pdf')
#     else:
#         plt.savefig("./results/" + exp_date + "/" + 'samplesize' + '.pdf', format='pdf')
# plt.show()
