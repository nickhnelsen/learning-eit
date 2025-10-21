import torch
import math

class RandomField(object):
    """       
    Samples N RFs with Matern covariance with periodic BCs only; supports spatial dimensions up to d=3.
    Supports Gaussian or Uniform KLE coefficients only.
    Requires PyTorch version >=1.8.0 (torch.fft).
    """
    def __init__(self,
                 size,
                 dim=2,
                 alpha=1.5,
                 tau=10,
                 sigma=None,
                 distribution="gaussian",
                 device=None
                 ):
        """
        Size must be even.
        """
        super().__init__()
        
        self.size = None    # to be assigned later
        self.dim = dim
        self.alpha = alpha
        self.tau = tau
        self.sigma = sigma
        self.distribution = distribution
        self.device = device
        
        if size % 2 != 0:
            raise ValueError("The argument 'size' must be even.")

        if self.sigma is None:
            self.sigma = self.tau**(0.5*(2*self.alpha - self.dim))
            
        k_max = size//2

        if self.dim == 1:
            k = torch.cat((torch.arange(start=0, end=k_max, step=1, device=self.device), \
                           torch.arange(start=-k_max, end=0, step=1, device=self.device)), 0)

            self.sqrt_eig = size*self.sigma*((4*(math.pi**2)*(k**2) + self.tau**2)**(-self.alpha/2.0))
            self.sqrt_eig[0] = 0.0

        elif self.dim == 2:
            wavenumbers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=self.device), \
                                    torch.arange(start=-k_max, end=0, step=1, device=self.device)), \
                                    0).repeat(size,1)

            wavenumbers = (wavenumbers.transpose(0,1))**2 + wavenumbers**2

            self.sqrt_eig = (size**2)*self.sigma*((4*(math.pi**2)*wavenumbers + \
                                                        self.tau**2)**(-self.alpha/2.0))
            self.sqrt_eig[0,0] = 0.0

        elif self.dim == 3:
            wavenumbers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=self.device), \
                                    torch.arange(start=-k_max, end=0, step=1, device=self.device)), \
                                    0).repeat(size,size,1)

            wavenumbers = (wavenumbers.transpose(1,2))**2 + wavenumbers**2 + (wavenumbers.transpose(0,2))**2

            self.sqrt_eig = (size**3)*self.sigma*((4*(math.pi**2)*wavenumbers + self.tau**2)**(-self.alpha/2.0))
            self.sqrt_eig[0,0,0] = 0.0

        self.size = []
        for j in range(self.dim):
            self.size.append(size)

        self.size = tuple(self.size)
        
        if self.distribution == "gaussian":
            self._get_coeff = self._get_gaussian
        elif self.distribution == "uniform":
            self._get_coeff = self._get_uniform
        else:
            raise ValueError("distribution must be either gaussian or uniform")
        
    def _get_gaussian(self, N):
        coeff = torch.randn(math.ceil(N/2), *self.size, 2, device=self.device)  # real & imag iid N(0,1)
        return coeff
    
    def _get_uniform(self, N):
        root3 = math.sqrt(3)
        coeff = torch.rand(math.ceil(N/2), *self.size, 2, device=self.device)
        coeff = 2*root3*coeff - root3   # real & imag iid U(-root3,root3)
        return coeff

    def sample(self, N):
        """
        Input:
            N: (int), number of GRF samples to return
        
        Output:
            u: (N, size, ..., size) tensor
        """
        u = self._get_coeff(N)
        u = self.sqrt_eig*(u[...,0] + u[...,1]*1.j) # complex KL expansion coefficients
        
        u = torch.fft.ifftn(u, s=self.size)                 # u_1 + u_2 i
        u = torch.cat((torch.real(u), torch.imag(u)), 0)
        
        if N % 2 == 0:
            return u
        else: # N is odd, drop one sample before returning
            return u[:-1,...] 

    def __call__(self, N):
        return self.sample(N)


if __name__=='__main__':
    
    import matplotlib.pyplot as plt
    
    sz = 512
    alpha = 1.5
    tau = 10
    distribution="uniform" # "uniform" or "gaussian"
    N = 4

    rf = RandomField(sz, alpha=alpha, tau=tau, distribution=distribution)
    samples = rf(N)
    
    for i, sample in enumerate(samples):
        plt.figure(i)
        plt.imshow(sample)
        plt.show()