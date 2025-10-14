% Computation of the Dirichlet-to-Neumann map as a matrix acting 
% in trigonometric basis. Omega is the unit disc.
%
% We actually precompute the Neumann-to-Dirichlet map using FEM,
% and then find DN map by inverting the ND map and adding 
% appropriate mapping of constant functions.
%
% We compute in addition the DN map related to the constant 
% conductivity 1 (analytically)
% and save both DN matrices to file data/ex2DN.mat.
% 
% Routine ND_GRF_comp.m must be run before this file.
%
% Produces regularized inverse problem solution map assuming noisy NtoD:
% noise \sim N(0, nug*Id), domain of R_gamma x-prior such that N(0,C)(.H^-1/2) = 1
%
% Nicholas H. Nelsen Jan. 2022

% Load precomputed data
load data/ND NtoD Nvec Ntrig nug

% NOTE: setup to construct prior with full support on \dot{H}^{-1/2} (domain of DN map)
beta = 1.1;     % beta > 0 (H^-1/2 a.s.) or beta > 1/2 (L^2 a.s.)
tauDN = 3;
 
% Invert the ND matrix to get DN matrix in trigonometric basis apart
% from constant functions.
cov = ((tauDN/(2*pi))^(2*beta-1))*(Nvec.^2 + (tauDN/(2*pi))^2).^(-beta);
DN = (NtoD'*NtoD + nug*diag(1./cov))\(NtoD');
DN = (DN + DN')/2;  % self-adjoint part

% Add appropriate zero row and zero column
DN = [DN(:,1:Ntrig),zeros(2*Ntrig,1),DN(:,Ntrig+1:end)];
DN = [DN(1:Ntrig,:);zeros(1,2*Ntrig+1);DN(Ntrig+1:end,:)];

% Compute DN map of unit conductivity (analytically)
DNvec = -Ntrig:Ntrig;
DN1 = abs(diag(DNvec));

% Save result to file
save data/ex2DN DN DN1 Ntrig beta tauDN