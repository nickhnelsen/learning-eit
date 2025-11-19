% Plot the result of ex2Kvec_comp.m
%
% Samuli Siltanen June 2012

% Load precomputed results
load('/media/nnelsen/SharedHDD2TB/datasets/eit/dbar/data/ex2Kvec.mat', 'Kvec', 'R');
% load data/ex2Kvec Kvec R

% Plot the points
figure(1)
clf
plot(real(Kvec),imag(Kvec),'r.','markersize',6)
axis equal
axis([-R R -R R])

%print -depsc ex2Kvec.png

