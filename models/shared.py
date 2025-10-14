import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F


def _get_act(act):
    """
    https://github.com/NeuralOperator/PINO/blob/master/models/utils.py
    """
    if act == 'tanh':
        func = F.tanh
    elif act == 'gelu':
        func = F.gelu
    elif act == 'relu':
        func = F.relu_
    elif act == 'elu':
        func = F.elu_
    elif act == 'leaky_relu':
        func = F.leaky_relu_
    else:
        raise ValueError(f'{act} is not supported')
    return func


class MLP(nn.Module):
    """
    Pointwise single hidden layer fully-connected neural network applied to last axis of input
    """
    def __init__(self, channels_in, channels_hid, channels_out, act='gelu'):
        super(MLP, self).__init__()
        
        self.fc1 = nn.Linear(channels_in, channels_hid)
        self.act = _get_act(act)
        self.fc2 = nn.Linear(channels_hid, channels_out)

    def forward(self, x):
        """
        Input shape (of x):     (..., channels_in)
        Output shape:           (..., channels_out)
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


def compl_mul(input_tensor, weights):
    """
    Complex multiplication:
    (batch, in_channel, ...), (in_channel, out_channel, ...) -> (batch, out_channel, ...), where ``...'' represents the spatial part of the input.
    """
    return torch.einsum("bi...,io...->bo...", input_tensor, weights)


################################################################
#
# 1d helpers
#
################################################################
def resize_rfft(ar, s):
    """
    Truncates or zero pads the highest frequencies of ``ar'' such that torch.fft.irfft(ar, n=s) is either an interpolation to a finer grid or a subsampling to a coarser grid.
    Args
        ar: (..., N) tensor, must satisfy real conjugate symmetry (not checked)
        s: (int), desired irfft output dimension >= 1
    Output
        out: (..., s//2 + 1) tensor
    """
    N = ar.shape[-1]
    s = s//2 + 1 if s >=1 else s//2
    if s >= N: # zero pad or leave alone
        out = torch.zeros(list(ar.shape[:-1]) + [s - N], dtype=torch.cfloat, device=ar.device)
        out = torch.cat((ar[..., :N], out), dim=-1)
    elif s >= 1: # truncate
        out = ar[..., :s]
    else: # edge case
        raise ValueError("s must be greater than or equal to 1.")

    return out


def resize_fft(ar, s):
    """
    Truncates or zero pads the highest frequencies of ``ar'' such that torch.fft.ifft(ar, n=s) is either an interpolation to a finer grid or a subsampling to a coarser grid.
    Reference: https://github.com/numpy/numpy/pull/7593
    Args
        ar: (..., N) tensor
        s: (int), desired ifft output dimension >= 1
    Output
        out: (..., s) tensor
    """
    N = ar.shape[-1]
    if s >= N: # zero pad or leave alone
        out = torch.zeros(list(ar.shape[:-1]) + [s - N], dtype=torch.cfloat, device=ar.device)
        out = torch.cat((ar[..., :N//2], out, ar[..., N//2:]), dim=-1)
    elif s >= 2: # truncate modes
        if s % 2: # odd
            out = torch.cat((ar[..., :s//2 + 1], ar[..., -s//2 + 1:]), dim=-1)
        else: # even
            out = torch.cat((ar[..., :s//2], ar[..., -s//2:]), dim=-1)
    else: # edge case s = 1
        if s < 1:
            raise ValueError("s must be greater than or equal to 1.")
        else:
            out = ar[..., 0:1]

    return out


################################################################
#
# 2d helpers
#
################################################################
def resize_rfft2(ar, s):
    """
    Truncates or zero pads the highest frequencies of ``ar'' such that torch.fft.irfft2(ar, s=s) is either an interpolation to a finer grid or a subsampling to a coarser grid.
    Args
        ar: (..., N_1, N_2) tensor, must satisfy real conjugate symmetry (not checked)
        s: (2) tuple, s=(s_1, s_2) desired irfft2 output dimension (s_i >=1)
    Output
        out: (..., s1, s_2//2 + 1) tensor
    """
    s1, s2 = s
    out = resize_rfft(ar, s2) # last axis (rfft)
    return resize_fft(out.transpose(-2,-1), s1).transpose(-2,-1) # second to last axis (fft)


def resize_fft2(ar, s):
    """
    Truncates or zero pads the highest frequencies of ``ar'' such that torch.fft.ifft2(ar, s=s) is either an interpolation to a finer grid or a subsampling to a coarser grid.
    Args
        ar: (n, c, N_1, N_2) tensor
        s: (2) tuple, s=(s_1, s_2) desired ifft2 output dimension (s_i >=1)
    Output
        out: (n, c, s1, s_2) tensor
    """
    s1, s2 = s
    out = resize_fft(ar, s2) # last axis (fft)
    return resize_fft(out.transpose(-2,-1), s1).transpose(-2,-1) # second to last axis (fft)
    
def get_grid2d(shape, device):
    """
    Returns a discretization of the 2D identity function on [0,1]^2
    """
    batchsize, size_x, size_y = shape[0], shape[1], shape[2]
    gridx = torch.linspace(0, 1, size_x)
    gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
    gridy = torch.linspace(0, 1, size_y)
    gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
    return torch.cat((gridx, gridy), dim=-1).to(device)


def projector2d(x, s=None):
    """
    Either truncate or zero pad the Fourier modes of x so that x has new resolution s (s is 2 tuple)
    """
    if s is not None and tuple(s) != tuple(x.shape[-2:]):
        x = fft.irfft2(resize_rfft2(fft.rfft2(x, norm="forward"), s), s=s, norm="forward")
        
    return x


def process_raw_ntd(ntd, s):
    """
    Adds missing zero Fourier modes to the raw data
    Truncates or zero pads the highest frequencies in last two axes
    Removes negative frequencies in the -2 axis
    Args
        ntd:    (n, c, N1 - 1, N2 - 1) tensor in fft2 format WITHOUT 0 wavenumbers
        s:      (2) tuple, s=(s_1, s_2) desired ifft2 output dimension (s_i >=1)
    Output
        out:    (n, c, s1//2 + 1, s_2) tensor
    """
    ntd = F.pad(ntd, [1, 0, 1, 0])
    ntd = resize_fft2(ntd, s)
    ntd = ntd[..., :(ntd.shape[-2] + 1)//2 + 1, :]
    return ntd

################################################################
#
# 2d Fourier layers
#
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        """
        Fourier integral operator layer defined for functions over the torus. Maps functions to functions.
        """
        super(SpectralConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1 
        self.modes2 = modes2

        self.scale = 1. / (self.in_channels * self.out_channels)
        self.weights1 = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def forward(self, x, s=None):
        """
        Input shape (of x):     (batch, channels, ..., nx_in, ny_in)
        s:                      (list or tuple, length 2): desired spatial resolution (s,s) in output space
        """
        # Original resolution
        out_ft = list(x.shape)
        out_ft[1] = self.out_channels
        xsize = out_ft[-2:]

        # Compute Fourier coeffcients (un-scaled)
        x = fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(*out_ft[:-2], xsize[-2], xsize[-1]//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[..., :self.modes1, :self.modes2] = \
            compl_mul(x[..., :self.modes1, :self.modes2], self.weights1)
        out_ft[..., -self.modes1:, :self.modes2] = \
            compl_mul(x[..., -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        if s is None or tuple(s) == tuple(xsize):
            x = fft.irfft2(out_ft, s=tuple(xsize))
        else:
            x = fft.irfft2(resize_rfft2(out_ft, s), s=s, norm="forward") / (xsize[-2] * xsize[-1])

        return x
