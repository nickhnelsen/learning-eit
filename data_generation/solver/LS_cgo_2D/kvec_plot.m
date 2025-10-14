% Load precomputed results
load data/kvec kvec R

% Plot the points
figure(1)
clf
plot(real(kvec),imag(kvec),'r.','markersize',6)
axis equal
axis([-R R -R R])

%print -depsc ex2Kvec.eps

