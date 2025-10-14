% Written by:
% Nicholas H. Nelsen
% California Institute of Technology
% Email: nnelsen@caltech.edu

% Set up for conductivity calculation; requires function file GRFcos.m.
% Creates a smoothed version of an originally discontinuous piecewise constant conductivity
% Default: tau=20, alpha=4.5, rho=0.75, contrast_ratio=30, sigma=25

% Last updated: Jan. 2023

function [raw_cond, interp_cond] = get_cond_data_smooth(N,tau,alpha,rho,contrast_ratio,sigma)
    g_plus = contrast_ratio;	
    g_minus = 1;

    % Sample
    g = GRFcos(N, tau, alpha);
    raw_cond = zeros(size(g));
    raw_cond(g >= 0) = g_plus;
    raw_cond(g < 0) = g_minus;

    % Return gridded interpolant
    gridvec = -1:2/(N-1):1;             % Grid vector for [-1,1] with N points
    [W1, W2] = meshgrid(gridvec);
    g = (abs(W1 + 1i*W2)>rho);
    raw_cond(g) = 1;  % Threshold to 1 in a neighborhood of disk bdd
    raw_cond = imgaussfilt(raw_cond, sigma); % Smooth the field with Gaussian conv with stdev sigma
    interp_cond = griddedInterpolant({gridvec, gridvec.'},raw_cond.','spline');
end