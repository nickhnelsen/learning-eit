% Written by:
% Nicholas H. Nelsen
% California Institute of Technology
% Email: nnelsen@caltech.edu

% Set up for conductivity calculation; requires function file GRFcos.m.
% Creates a truncated lognormal conductivity
% Default: tau=8, alpha=3.1, Rp=0.95, Rm=0.5, scale=8

% Last updated: Jan. 2023

function [g, interp_cond] = get_cond_data_lognormaltrunc(N,tau,alpha,Rp,Rm,scale)
    % Sample
    g = exp(GRFcos(N, tau, alpha));

    % Return gridded interpolant
    gridvec = -1:2/(N-1):1;             % Grid vector for [-1,1] with N points
    [W1, W2] = meshgrid(gridvec);
    r = abs(W1 + 1i*W2);
    idx_big = (r>Rp);
    gmin = 1.0;                         % Threshold to 1 near disk bdd ( e^mean = 1, mean(g)=0)
    g(idx_big) = gmin;
    idx = (r<Rp) & (r>Rm);              % medium indices
    cut = cutoff(r(idx), gmin./g(idx), Rm, Rp, scale);
    g(idx) = g(idx).*cut;
    interp_cond = griddedInterpolant({gridvec, gridvec.'},g.','spline');
end