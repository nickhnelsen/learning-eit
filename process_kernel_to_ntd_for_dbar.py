import torch
from util.sample_random_fields import RandomField
from util.utilities_module import integrate

import math
from scipy.io import savemat

torch.set_printoptions(precision=16)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device is", device)



def my_flip(x):
    x = torch.cat((x[...,0:1], torch.flip(x[...,1:], [-1])), -1)
    return x

def get_ntd_from_kernel(x, shift=True):
    """
    Return in ascending frequency order: -N+1, -N+2, -1, 1, ..., N

    x: (N, J, J) real tensor of kernel function values on J by J grid of [0,2pi]^2

    Returns: (N, J-1, J-1) complex tensor representing NtD matrix in Fourier coordinates, with (0,0) constant Fourier mode removed
    """
    x = torch.fft.fft2(2*math.pi * x, norm="forward")
    x = my_flip(x) # (N, J, J) complex tensor in fft2 format
    if shift:
        x = torch.fft.fftshift(x, dim=(-2,-1)) # (N, J, J) in ascending order in last two dims
        xdims = x.shape[-2:]
    
        J = xdims[-2]
        mask2 = torch.ones(J, dtype=bool)
        mask2[J//2] = False
        
        J = xdims[-1]
        mask1 = torch.ones(J, dtype=bool)
        mask1[J//2] = False
        
        x = x[..., mask2, :][..., :, mask1] # remove (0,0) mode
    else:
        x = x[..., 1:, 1:] # remove (0,0) mode
    return x



save_path = "/media/nnelsen/SharedHDD2TB/datasets/eit/dbar/ntd_samples/"
Ntrig = 128

N_train = 9500
noise_new = 1
noise_distribution_new = "uniform"

N_val = 100
N_test = 400
N_max = 10000

Nvec = torch.cat([torch.arange(-Ntrig + 1, 0), torch.arange(1, Ntrig+1)]).to(torch.float64)

# three conductitivities: 1) OOD shape detection heart lungs 2) x_test[0,...] shape, 3) heart_lung three phase
def save_ntd(kernel, Ntrig=Ntrig, Nvec=Nvec, shift=True, pathname='ND.mat'):
    assert kernel.shape[-1] == 2*Ntrig
    
    ntd = get_ntd_from_kernel(kernel.to(torch.float64), shift=shift).squeeze()
    
    savemat(save_path + pathname, {
        'NtoD': ntd.detach().cpu().numpy(),
        'Nvec': Nvec.detach().cpu().numpy(),
        'Ntrig': Ntrig,
    })
    return ntd
    
# Get noisy inputs
def get_noisy(dataset, my_noise=noise_new, my_noise_distribution=noise_distribution_new):
    rf = RandomField(dataset.shape[-1], distribution=my_noise_distribution, device=device)
    dataset_noisy = rf.generate_noise_dataset(dataset.shape[0])
    dataset_noisy = (my_noise/100)*(integrate(dataset**2).sqrt()[:,None,None])*dataset_noisy
    dataset_noisy = dataset + dataset_noisy
    return dataset_noisy

# 3
load_path = "/home/nnelsen/code/learning-eit/data_generation/solver_heartNlungs/data/"

kernel = torch.load(load_path + 'kernel_heart.pt', weights_only=True)['kernel_3heart'][0,...][None,...]
kernel_noisy = get_noisy(kernel, noise_new, noise_distribution_new)

ntd_3_clean = save_ntd(kernel, pathname='ND_heart_three_phase_clean.mat')
ntd_3_noisy = save_ntd(kernel_noisy, pathname='ND_heart_three_phase_noisy.mat')

################################################################
#
# load and process bigger data
#
################################################################

data_folder = '/media/nnelsen/SharedHDD2TB/datasets/eit/'

x_test3 = torch.load(data_folder + 'kernel_3heart_rhop7.pt', weights_only=True)['kernel_3heart'][...,::2,::2]
x_test3 = x_test3[0,...].unsqueeze(0)
x_test_clean = torch.load(data_folder + 'kernel.pt', weights_only=True)['kernel'][...,::2,::2]

# Fix same test data for all experiments
x_test_clean = x_test_clean[-(N_val + N_test):,...]
x_test_clean = x_test_clean[-N_test:,...]
x_test_clean = x_test_clean[0,...].unsqueeze(0)
x_test3_clean = x_test3[...]


x_test3 = get_noisy(x_test3_clean, noise_new, noise_distribution_new)
x_test = get_noisy(x_test_clean, noise_new, noise_distribution_new)

ntd_1_clean = save_ntd(x_test3_clean, pathname='ND_heart_shape_clean.mat')
ntd_1_noisy = save_ntd(x_test3, pathname='ND_heart_shape_noisy.mat')

ntd_2_clean = save_ntd(x_test_clean, pathname='ND_idx0_shape_clean.mat')
ntd_2_noisy = save_ntd(x_test, pathname='ND_idx0_shape_noisy.mat')
