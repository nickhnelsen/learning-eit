% Dbar solver

clc; clear variables; close all;

%% precompute if needed
FLAG_PRE = true;
if FLAG_PRE
    precompute;
end

%% get DtN map from NtD
ex2DN_comp;

%% CGO traces
ex2psi_BIE_comp;

%% Scattering transform k\mapsto t(k), \abs(k)\leq R
ex2tBIE_comp;
% ex2tBIE_plot; % TODO: Check plot of scattering transform here, estimate truncation radius of disk based on noisyness

%% Solve Dbar to get full CGO solution and hence conductivity via point evaluation at k=0
ex2tBIErecon_comp;
ex2tBIErecon_plot;
