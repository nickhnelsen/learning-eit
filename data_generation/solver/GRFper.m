% Written by:
% Nicholas H. Nelsen
% California Institute of Technology
% Email: nnelsen@caltech.edu

% Return a sample of a Gaussian random field on the torus [0,2*pi]_{per} (L=2pi) with: 
%       mean function m = 0
%       covariance operator C = (tau/2pi)^(2*alpha-1)*(-Delta + (tau/2pi)^2*I)^(-alpha),
%       (normalization for complex Fourier basis on [a,b]^d is (b-a)^-d/2)
% where Delta is the Laplacian on the torus [0,2*pi]_{per}.

% Last updated: Aug. 2022

function U = GRFper(N, tau, alpha)
% Input
%   N:      (int)   grid size, must be even
%   tau:    (float) inverse lengthscale for Gaussian measure covariance operator
%   al:     (float) regularity of covariance operator
% Output
%   U: (N,) vector of Gaussian random field on torus [0,2*pi]_{per}
	
    % Check even grid size
    assert(mod(N,2)==0);
    kmax = fix(N/2);
    
	% Complex N^c(0,1) iid Gaussian RVs
    xi = normrnd(0,1,kmax+1,2)/sqrt(2);     % wavenumbers k=0,1,...,kmax
    xi = xi(:,1) + 1i*xi(:,2);

    % Define the (square root of) eigenvalues of the covariance operator
    k = (0:kmax).';
	coef = (tau^(alpha-1/2))*((2*pi*k).^2 + tau^2).^(-alpha/2);
    
	% Construct the KL (discrete Fourier transform) coefficients
	B = N*coef.*xi;                 % multiply by N to satisfy MATLAB definition of ifft
	B = [B;conj(B(kmax:-1:2))];     % real conjugate symmetry
    B(1) = 0;                       % ignore k=0 constant mode to enforce zero mean samples
    % B(kmax+1) = 0;                % (easy choice) zero out Nyquist mode
    B(kmax+1) = 2*real(B(kmax+1));  % require B(kmax+1) = 2*real(B(kmax+1)) because U is real

    % Inverse (fast FFT-based) discrete Fourier transform
    U = ifft(B,'symmetric');
end