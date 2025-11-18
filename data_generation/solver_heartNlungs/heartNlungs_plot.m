% Plot the conductivity 
%
% Samuli Siltanen May 2012

% Set font size
fsize = 14;

% Create evaluation points
t       = linspace(-1,1,256);
[x1,x2] = meshgrid(t);
z       = x1 + 1i*x2;

% Evaluate potential
c = heartNlungs(z);
save data/cond_heartNlungs c z
c(abs(z)>1) = NaN;

% Two-dimensional plot 
figure(2)
clf
imagesc(c,[0.3,7])
colormap parula
map = colormap;
colormap([[1 1 1];map]);
axis equal
axis off
colorbar
%print -dpng heartNlungs2D.png
