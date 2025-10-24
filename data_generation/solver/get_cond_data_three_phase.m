% Written by:
% Nicholas H. Nelsen
% California Institute of Technology
% Email: nnelsen@caltech.edu

% Set up for conductivity calculation; requires function file GRFcos.m
% Vector level set method

function [raw_cond, interp_cond] = get_cond_data_three_phase(N,tau,alpha,rho,val1,val2)
    % Sample
    g_1 = GRFcos(N, tau, alpha);
    g_2 = GRFcos(N, tau, alpha);
    
%     region1 = (g_1 >= 0) & (g_2 < 0);
%     region2 = (g_1 < 0) & (g_2 >= 0);
%     region0 = ~(region1 | region2);

    region1 = (g_1 >= 0) & (g_2 >= 0);
    region2 = (g_1 >= 0) & (g_2 < 0);
    region0 = (g_1 < 0);
    
    raw_cond = zeros(size(g_1));
    raw_cond(region1) = val1;
    raw_cond(region2) = val2;
    raw_cond(region0) = 1;

    % Gridded interpolant
    gridvec = -1:2/(N-1):1;             % Grid vector for [-1,1] with N points
    [W1, W2] = meshgrid(gridvec);
    raw_cond(abs(W1 + 1i*W2)>rho) = 1;  % Threshold to 1 in a neighborhood of disk bdd
    interp_cond = griddedInterpolant({gridvec, gridvec.'},raw_cond.','nearest');
end