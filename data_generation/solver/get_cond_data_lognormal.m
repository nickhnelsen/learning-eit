% Written by:
% Nicholas H. Nelsen
% California Institute of Technology
% Email: nnelsen@caltech.edu

% Set up for conductivity calculation; requires function file GRFcos.m.
% Creates a lognormal conductivity
% Default: tau=10 and alpha=3

% Last updated: Jan. 2023

function [g, interp_cond] = get_cond_data_lognormal(N,tau,alpha)
    % Sample
    g = exp(GRFcos(N, tau, alpha));

    % Return gridded interpolant
    gridvec = -1:2/(N-1):1;             % Grid vector for [-1,1] with N points
    interp_cond = griddedInterpolant({gridvec, gridvec.'},g.','spline');
end