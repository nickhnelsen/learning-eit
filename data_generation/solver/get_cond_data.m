% Written by:
% Nicholas H. Nelsen
% California Institute of Technology
% Email: nnelsen@caltech.edu

% Set up for conductivity calculation; requires function file GRFcos.m

% Last updated: Jan. 2022

function [raw_cond, interp_cond] = get_cond_data(N,tau,alpha,rho,contrast_ratio)
    % TODO: can later include multiple level sets instead of just one
    g_plus = contrast_ratio;	
    g_minus = 1;

    % Sample
    g = GRFcos(N, tau, alpha);
    raw_cond = zeros(size(g));
    raw_cond(g >= 0) = g_plus;
    raw_cond(g < 0) = g_minus;

    % Gridded interpolant
    gridvec = -1:2/(N-1):1;             % Grid vector for [-1,1] with N points
    [W1, W2] = meshgrid(gridvec);
    raw_cond(abs(W1 + 1i*W2)>rho) = 1;  % Threshold to 1 in a neighborhood of disk bdd
    interp_cond = griddedInterpolant({gridvec, gridvec.'},raw_cond.','nearest');
end