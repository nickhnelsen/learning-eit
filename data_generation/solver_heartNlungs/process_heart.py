import torch
import torch.nn.functional as F
import math
import os, sys; sys.path.append(os.path.join('../..'))
from util.utilities_module import MatReader


def get_kernel_from_ntd(x):
    """
    x: (N, J-1, J-1) complex tensor representing NtD matrix in Fourier coordinates
    
    Returns: (N, J, J) real tensor of kernel function values on J by J grid of [0,2pi]^2
    """
    x = F.pad(torch.flip(x, [-1]), (1, 0, 1, 0, 0, 0))
    x = torch.real(torch.fft.ifft2(x, norm="forward"))/(2*math.pi)
    return x


data_folder = './data/'
FLAG_SAVE = True

conductivity = []
kernel = []

x = MatReader(data_folder + "ND.mat", variable_names=['NtoD'])
kernel = x.read_field('NtoD')[None, ...]
kernel = get_kernel_from_ntd(kernel).repeat(3,1,1)

y = MatReader(data_folder + "cond_heartNlungs.mat", variable_names=['c'])
conductivity = y.read_field('c')[None, ...].repeat(3,1,1)

if FLAG_SAVE:
    torch.save({'conductivity_3heart': conductivity}, data_folder + 'conductivity_heart.pt')
    torch.save({'kernel_3heart': kernel}, data_folder + 'kernel_heart.pt')


# %% Plot
from util import plt

plot_ind = 0

plt.close("all")

X1 = torch.linspace(-1, 1, conductivity.shape[-1])
X1, Y1 = torch.meshgrid(X1, X1)

plt.figure(0)
cplot = torch.clone(conductivity[plot_ind,...])
mask_out = ~(torch.abs(X1 + 1j * Y1) <= 1)
mask_in = ~mask_out
cplot[mask_out] = float('nan')
plt.imshow(cplot, origin='lower',extent=[-1, 1, -1, 1])
plt.show()

plt.figure(1)
plt.imshow(kernel[plot_ind,...], origin='lower',extent=[0, 2*math.pi, 0, 2*math.pi])
plt.show()
