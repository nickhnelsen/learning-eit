% Plot the results of ex2tBIErecon_comp.m
%
% Samuli Siltanen June 2012; Nicholas H. Nelsen 2025

% Multiple of upsampling the reconstruction (see below)
% USmultiple = 8;

% Load precomputed reconstruction and its evaluation points
load('/media/nnelsen/SharedHDD2TB/datasets/eit/dbar/data/ex2recon.mat', 'x1', 'x2', 'recon');
recon = reshape(real(recon),size(x1));

szz = length(recon);
% Create evaluation points
t       = linspace(-1,1,szz);
[x1,x2] = meshgrid(t);
z       = x1 + 1i*x2;

% Evaluate potential
recon(abs(z)>1) = NaN;

% Two-dimensional plot 
figure(2)
clf
imagesc(recon, 'AlphaData', ~isnan(recon));
colormap(parula)
axis equal
axis off
set(gca, 'color', [1 1 1]);   % white background behind transparent NaNs
colorbar

% Write image to file
% print -dpng ex2recon.png