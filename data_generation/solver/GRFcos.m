% Written by:
% Nicholas H. Nelsen
% California Institute of Technology
% Email: nnelsen@caltech.edu

% NHN modified code, original by Matt Dunlop
% Return a sample of a Gaussian random field on [-1,1]^2 (L=1-(-1)=2) with: 
%       mean function m = 0
%       covariance operator C = (tau/2)^(2*alpha-2)*(-Delta + (tau/2)^2*I)^(-alpha),
%       (normalization for cosine basis on [a,b]^d is (b-a)^-d/2)
% where Delta is the Laplacian with zero Neumann boundary conditions.

% Last updated: Aug. 2022

function U = GRFcos(N, tau, alpha)
% Input
%   N:      (int)   grid size in one dimension
%   tau:    (float) inverse lengthscale for Gaussian measure covariance operator
%   al:     (float) regularity of covariance operator
% Output
%   U: (N,N) matrix of Gaussian random field on the grid meshgrid(-1:2/(N-1):1,-1:2/(N-1):1)
	
	% (N,N) matrix of N(0,1) iid Gaussian RVs
    xi = normrnd(0,1,N);          

    % Define the (square root of) eigenvalues of the covariance operator
	[K1,K2] = meshgrid(0:N-1);    % eig index k = (K1,K2)
	coef = (tau^(alpha-1))*((pi^2)*(K1.^2 + K2.^2) + tau^2).^(-alpha/2);
    
	% Construct the KL (discrete cosine transform) coefficients
	B = N*coef.*xi;             % multiply by N to satisfy MATLAB definition of IDCT2
    B(1,:) = sqrt(2)*B(1,:);    % adjust for wavenumber 0
    B(:,1) = sqrt(2)*B(:,1);    % adjust for wavenumber 0
    B(1,1) = 0;                 % ignore k=(0,0) constant mode
	
    % Inverse (fast FFT-based) 2D discrete cosine transform
    U = idct2(B);               % sums B*2/N*cos(k1 pi z1)cos(k2 pi z2) over k1,k2; z1,z2\in(0,1)
    
    % Interpolate to physical grid x \in [-1,1]^2 containing the boundary
    [X1,Y1] = meshgrid(1/(2*N):1/N:(2*N-1)/(2*N));    % IDCT grid
	[X2,Y2] = meshgrid(0:1/(N-1):1);    % physical grid in terms of z=(x-(-1))/(1-(-1)) 
    U = interp2(X1,Y1,U,X2,Y2,'spline');
end