% This routine is for computing the scattering transform at the k-grid
% created by routine ex2Kvec_comp.m (and saved to file data/ex2Kvec.mat).
%
% We load from file the Fourier coefficients of the traces of Faddeev 
% exponentially growing solutions created by routine ex2psi_BIE_comp.m 
% (and saved to file data/ex2psi_BIE.mat).
% The traces are evaluated at the following points on the unit circle:
% exp(i\theta), where vector \theta is previously saved to file data/theta.mat.
%
% Then we integrate according to the definition 
%
%  tBIE(k) = \int_{boundary} e^{i*conj(kx)} (Lg-L1) psi(x,k) d\sigma(x)
%
% using a loop over all k values in the vector Kvec.
% The result is saved to file data/ex2tBIE.mat.
%
% Samuli Siltanen June 2012; Nicholas H. Nelsen 2025

% Load vectors of k and theta values
load('/media/nnelsen/SharedHDD2TB/datasets/eit/dbar/data/ex2Kvec.mat', 'Kvec');
% load data/ex2Kvec Kvec
load('/media/nnelsen/SharedHDD2TB/datasets/eit/dbar/data/theta.mat', 'theta', 'Ntheta', 'Dtheta');
% load data/theta theta Ntheta Dtheta

% Load precomputed Fourier coefficients of the traces of psi
load('/media/nnelsen/SharedHDD2TB/datasets/eit/dbar/data/ex2psi_BIE.mat', 'Fpsi_BIE', 'Nvec');
% load  data/ex2psi_BIE Fpsi_BIE

% Load precomputed DN maps
load('/media/nnelsen/SharedHDD2TB/datasets/eit/dbar/data/ex2DN.mat', 'DN', 'DN1','Ntrig');
% load data/ex2DN DN DN1 Ntrig
% Nvec = [-Ntrig+1 : Ntrig];

nnv = length(Nvec);

% Apply DN maps to evaluate (Lg-L1) psi(x,k) using memory efficient loop
FLLpsi = (DN - DN1)*Fpsi_BIE;
FLLpsi = reshape(FLLpsi.', [], 1, nnv);     % (n_k, 1, n_vec)
tBIE = exp(1i*reshape(theta*Nvec, 1, [], nnv)); % (1, n_theta, n_vec)
tBIE = squeeze(tBIE);
n_k = length(Kvec);
LLpsi  = zeros(n_k, Ntheta);
for k = 1:n_k
    DN = squeeze(FLLpsi(k,1,:)); % n_vec×1
    LLpsi(k,:) = (DN.' * tBIE.');   % (1×n_theta)
end
% Out of memory!
% LLpsi = sum(FLLpsi.*exp(1i*reshape(theta*Nvec, 1, [], nnv)), 3);

% Integrate
tBIE = sum(Dtheta*exp(1i*conj(Kvec)*exp(-1i*theta.')).*LLpsi, 2);

disp('Done (Scattering transform)')

% Save the result to file.
% load data/ex2Kvec Kvec tt tMAX
load('/media/nnelsen/SharedHDD2TB/datasets/eit/dbar/data/ex2Kvec.mat', 'Kvec', 'tt', 'tMAX');
save('/media/nnelsen/SharedHDD2TB/datasets/eit/dbar/data/ex2tBIE.mat', 'tBIE', 'Kvec', 'tt', 'tMAX');
% save data/ex2tBIE tBIE Kvec tt tMAX