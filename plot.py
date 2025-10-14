import torch
import numpy as np
import os
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

N_list = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 9500]
Noise_list = [0]
Seed_list = [0]
exp_date = "2025-10-14"
load_prefix = "paper_loop_debug"


# Legend
legs = [r"$p=1$", r"$p=2$"]

# Colors
color_list = ['k', 'C3', 'C5', 'C1', 'C2', 'C0', 'C4', 'C6', 'C7', 'C8', 'C9'] # black, red, brown, orange, green, blue, purple, pink, gray, olive, cyan
    
if FLAG_WIDE:
    plt.rcParams['figure.figsize'] = [6.0, 4.0]     # [6.0, 4.0]
else:
    plt.rcParams['figure.figsize'] = [6.0, 6.0]     # [6.0, 4.0]


plot_folder_base = "./results/" + exp_date + "/" + load_prefix
plot_errors = np.zeros((len(N_list), 2))
for i, N in enumerate(N_list):
    for Noise in Noise_list:
        for Seed in Seed_list:
            plot_folder = plot_folder_base + "_N" + str(N) + "_Noise" + str(Noise) + "_Seed" + str(Seed) + "/"
            # os.makedirs(plot_folder, exist_ok=True)

            # Load
            plot_errors[i,...] = torch.load(plot_folder + 'test_errors.pt', weights_only=True).numpy()
            
# errors = get_stats(errors_loops)
# errors_seq = get_stats(errors_seq_loops)
# errors_static = get_stats(errors_static_loops)
# errors_bary = get_stats(errors_bary_loops)
# errors_gmm = get_stats(errors_gmm_loops)
# errors_unif = get_stats(errors_unif_loops)



# Least square fit to error data
nplot = N_list[SHIFT:]
nplota = N_list
linefit = polyfit(np.log2(nplot), np.log2(plot_errors[SHIFT:,...]), 1)
lineplot = linefit[0,...] + linefit[1,...]*np.log2(nplot)[:,None]
lineplota = linefit[0,...] + linefit[1,...]*np.log2(nplota)[:,None]
my_slopes = -linefit[-1]
print("Least square slope fit is: ")
print(my_slopes)
np.save("./results/" + exp_date + "/" + 'rate_ls.npy', -linefit[-1])

# Experimental rates of convergence table
eocBoch = np.zeros([len(N_list)-1, 2])
for i in range(len(eocBoch)):
    eocBoch[i,...] = np.log2(plot_errors[i,...]/plot_errors[i + 1,...])/np.log2(N_list[i + 1]/N_list[i])
print("\nEOC is: ")
print(eocBoch)
np.save("./results/" + exp_date + "/" + "rate_table.npy", eocBoch)


# Plot: Err vs Sample size
plt.figure(0)

round_slopes = np.round(my_slopes, 2)
plt.loglog(N_list, 0.9*2**lineplota[...,0], ls='--', color='darkgray', label=fr'$N^{{-{round_slopes[0]:.2f}}}$')
plt.loglog(N_list, 1.07*2**lineplota[...,1], ls='-', color='darkgray', label=fr'$N^{{-{round_slopes[1]:.2f}}}$')

plot2_tup = [plot_errors[...,0], plot_errors[..., 1]]
for i, error_array in enumerate(plot2_tup):
    # twosigma = n_std*error_array[...,1]
    # lb = np.maximum(error_array[...,0] - twosigma, plot_tol)
    # ub = error_array[...,0] + twosigma

    plt.loglog(N_list, error_array, ls=style_list[i], color=color_list[i], marker=marker_list[i], markersize=msz, label=legs[i])
    
    # plt.fill_between(sample_size_list, lb, ub, facecolor=color_list[i], alpha=0.125)

plt.xlim(left=9e0)
plt.xlabel(r'Sample Size')
plt.ylabel(r'Average Relative $L^p$ Test Error')
plt.legend(framealpha=1, loc='best', borderpad=borderpad,handlelength=handlelength).set_draggable(True)
plt.grid(True, which="both")
plt.tight_layout()
if FLAG_save_plots:
    plt.savefig("./results/" + exp_date + "/" + 'samplesize' + '.pdf', format='pdf')
plt.show()
