% Written by:
% Nicholas H. Nelsen
% California Institute of Technology
% Email: nnelsen@caltech.edu

% Forward map run

% Last updated: Dec. 09, 2021

clc; clear variables; close all;

%% CGO traces
ex2psi_BIE_comp;

%% Scattering transform k\mapsto t(k), \abs(k)\leq R
ex2tBIE_comp;
ex2tBIE_plot;

%% Solve Dbar to get full CGO solution and hence conductivity via point evaluation at k=0
ex2tBIErecon_comp;
ex2tBIErecon_plot;