% Computes and saves the operator H_k acting on the Fourier basis 
% for a collection of grid points in the k-plane. 
% H_k is the following integral operator:
%
%   (H_k f)(x) = \int_{unitcircle} Htilde(k*(x-y))*f(y) dsigma(y), 
%
% where x is a point on the unit circle, 
% Htilde(x) = H1(x)-H1(0) = G1(x)-G0(x)-H1(0),
% and G1 is Faddeev's Green function for the Laplacian.
% The function Htilde can be evaluated using routine Htilde.m.
% 
% Dimension of trigonometric approximation is taken from file ex2DN_comp.m.
% Collection of grid points in the k-plane is created by ex2Kvec_comp.m.
%
% Samuli Siltanen June 2012; Nicholas H. Nelsen Dec. 2021

% NOTE: Order Ntrig of trigonometric approximation is loaded from file ex2DN_comp.m. 
% Basis will be exp(i*n*theta) for n = [-Ntrig : Ntrig].
load data/ND Ntrig

% Construct integration points (angles) on the circle
% NOTE: change unit circle discretization resolution here
Ntheta = 128;
theta  = 2*pi*[0:(Ntheta-1)]/Ntheta;
theta  = theta(:);
Dtheta = theta(2)-theta(1);
save data/theta theta Ntheta Dtheta

% Loop over points in the k-grid
load data/ex2Kvec Kvec
Nvec = [-Ntrig : Ntrig];
% NOTE: even faster acceleration with zero-padded FFT/IFFT up to Ntheta resolution
for kkk = 1:length(Kvec)
    k = Kvec(kkk);
    tic
    
    % Loop over Fourier basis functions and apply the operator to each of them.
    % Then compute inner products of the result with basis functions.
    % This way we build a matrix for the operator H_k.
    HH = H1tilde(k*(exp(1i*theta).' - exp(1i*theta))); % kernel function of intergal operator

    % Compute result of operator H_k applied to the basis function
    bfun = exp(1i*theta*Nvec); 
    Hk_bfun = Dtheta*(HH.')*bfun;
    
    % Project the result in trigonometric basis.
    Hk = 1/(2*pi)*Dtheta*exp(-1i*Nvec.'*theta.')*Hk_bfun;
    
    % Save the Hk matrix to file
    savecommand = ['save data/ex2Hk_', num2str(kkk), ' Hk'];
    eval(savecommand)
    disp(['Computation ', num2str(kkk), ' of ', num2str(length(Kvec)), ' took ', num2str(toc), ' seconds'])
end