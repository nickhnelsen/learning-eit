% Plot the result of the routines tLS_comp.m
clear variables; close all;

% Plot parameters
lwidth = 2;
fsize  = 16;
FLAG_SAVE = false;

% Load precomputed data
load data/tLS tLS
load data/kvec kvec R K1 K2

% Find index vector for points inside a disc
discind = (abs(K1+1i*K2)<R);

tmp = zeros(size(K1));
tmp(discind) = tLS;
tLS_plot1 = real(tmp);
tLS_plot2 = imag(tmp);

% Create customized colormap for white background 
colormap parula
MAP = colormap;
M = size(MAP,1); % Number of rows in the colormap
bckgrnd = [1 1 1]; % Pure white color
MAP = [bckgrnd;MAP];

% Modify the function for constructing white background
MIN = min(min([tLS_plot1,tLS_plot2]));
MAX = max(max([tLS_plot1,tLS_plot2]));
cstep = (MAX-MIN)/(M-1); % Step size in the colorscale from min to max
tLS_plot1(~discind) = MIN-cstep;
tLS_plot2(~discind) = MIN-cstep;

% Plot real and imaginary part of the scattering transform
figure(1)
clf
imagesc([tLS_plot1,tLS_plot2])
axis equal
axis off
colormap(MAP)
colorbar
title('Real part (left) and imaginary part (right)');

% Save image to file
if FLAG_SAVE
print -dpng tLS_complex_2D.png
end
