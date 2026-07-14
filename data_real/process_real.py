import torch
import torch.nn.functional as F
import math
import os, sys; sys.path.append(os.path.join('..'))
from util.utilities_module import MatReader
from models.shared import resize_fft2


def get_kernel_from_ntd(x, J=256):
    """
    x: (N, K-1, K-1) complex tensor representing NtD matrix in Fourier coordinates
    
    Returns: (N, J, J) real tensor of kernel function values on J by J grid of [0,2pi]^2
    """
    # Add zero modes with zero values
    x = F.pad(torch.flip(x, [-1]), (1, 0, 1, 0, 0, 0))
    
    # Zero pad to fine grid (if K<J, otherwise truncate to J modes)
    x = resize_fft2(x, s=(J,J))
    
    # Expand kernel onto grid [0,2\pi]^2_{per}
    x = torch.real(torch.fft.ifft2(x, norm="forward"))/(2*math.pi)
    return x


data_folder = './'
FLAG_SAVE = True

datapath = data_folder + "NtDmaps_all.mat"
x = MatReader(datapath, variable_names=['ntd_array'])
kernel = get_kernel_from_ntd(x.read_field('ntd_array'))

if FLAG_SAVE:
    torch.save({'kernel': kernel}, data_folder + 'kernel_KIT4.pt')


# %% Plot
from util import plt

plot_ind = 0

plt.close("all")

X1 = torch.linspace(-1, 1, kernel.shape[-1])
X1, Y1 = torch.meshgrid(X1, X1)

plt.figure(1)
plt.imshow(kernel[plot_ind,...], origin='lower', extent=[0, 2*math.pi, 0, 2*math.pi])
plt.show()
