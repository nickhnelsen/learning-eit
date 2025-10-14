% Written by:
% Nicholas H. Nelsen
% California Institute of Technology
% Email: nnelsen@caltech.edu

% Set up for conductivity calculation

% Last updated: Dec. 01, 2021


% Level set model
rho = 0.80;                         % Outer truncation radius to conductivity value 1 
contrast_ratio = 10;                % Max divided by min conductivity

% Random field model
tau = 20;
alpha = 4.5;

% Sample GRF resolution
N = 256;

% Initialize (can later include multiple level sets instead of just one)
g_plus = contrast_ratio;	
g_minus  = 1;

% Sample
g = GRFcos(N, tau, alpha);
Tg = zeros(size(g));
Tg(g >= 0) = g_plus;
Tg(g < 0) = g_minus;

% Gridded interpolant
gridvec = -1:2/(N-1):1;             % Grid vector for [-1,1] with N points
[W1, W2] = meshgrid(gridvec);
Tg(abs(W1 + 1i*W2)>rho) = 1;        % Threshold to 1 in a neighborhood of the boundary of disk
TgFunc = griddedInterpolant({gridvec, gridvec.'},Tg.','nearest');

% Save result to file
save data/single_conductivity TgFunc