import torch
import torch.nn.functional as F
import math
import os, sys; sys.path.append(os.path.join('..'))
from util.utilities_module import MatReader


def get_kernel_from_ntd(x):
    """
    x: (N, J-1, J-1) complex tensor representing NtD matrix in Fourier coordinates
    
    Returns: (N, J, J) real tensor of kernel function values on J by J grid of [0,2pi]^2
    """
    x = F.pad(torch.flip(x, [-1]), (1, 0, 1, 0, 0, 0))
    x = torch.real(torch.fft.ifft2(x, norm="forward"))/(2*math.pi)
    return x


data_folder = '/media/nnelsen/SharedHDD2TB/datasets/eit/lognormal_raw/'
N_loop = 40
FLAG_SAVE = not True

conductivity = []
kernel = []
for i in range(N_loop):
    datapath = data_folder + "eit_bin_fND_lognormal_S2025" + str(i + 1) + "_No250_Ni256_Ro256.mat"
    x = MatReader(datapath, variable_names=['cond_array','ntd_array'])
    conductivity.append(x.read_field('cond_array'))
    kernel.append(get_kernel_from_ntd(x.read_field('ntd_array')))

conductivity = torch.cat(conductivity, dim=0)
kernel = torch.cat(kernel, dim=0)

if FLAG_SAVE:
    torch.save({'conductivity': conductivity}, data_folder + 'conductivity.pt')
    torch.save({'kernel': kernel}, data_folder + 'kernel.pt')



# %% Plot
# from util import plt

# plot_ind = 0

# plt.close("all")

# X1 = torch.linspace(-1, 1, conductivity.shape[-1])
# X1, Y1 = torch.meshgrid(X1, X1)
# # mask = (torch.abs(X1 + 1j * Y1) <= 1.0)
# # # torch.save({'mask': mask}, data_folder + 'mask.pt')

# plt.figure(0)
# cplot = torch.clone(conductivity[plot_ind,...])
# mask_out = ~(torch.abs(X1 + 1j * Y1) <= 1)
# mask_in = ~mask_out
# cplot[mask_out] = float('nan')
# # cplot = mask_out + cplot*mask_in # set model to one outside unit disk radius 1
# plt.imshow(cplot, origin='lower',extent=[-1, 1, -1, 1])
# plt.show()

# plt.figure(1)
# plt.imshow(kernel[plot_ind,...], origin='lower',extent=[0, 2*math.pi, 0, 2*math.pi])
# plt.show()
