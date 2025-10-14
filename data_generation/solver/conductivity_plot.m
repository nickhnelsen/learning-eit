% Plot the single random conductivity 

% Set font size
fsize = 14;

% Create evaluation points
t       = linspace(-1,1,512);
[x1,x2] = meshgrid(t);
z       = x1 + 1i*x2;

% Evaluate conductivity
c = conductivity(z);
c(abs(z)>1) = NaN;

% Two-dimensional plot 
% Ref: https://blogs.helsinki.fi/smsiltan/2012/05/10/displaying-image-data-for-comparison/
figure(1)
clf
colormap parula
map = colormap;
M = size(map,1);                % Number of rows in the colormap
MAX = max(c(:));
MIN = min(c(:));
cstep = (MAX-MIN)/(M-1);        % Step size in the colorscale from min to max
c(abs(z)>1) = MIN - cstep;
imagesc(c)
colormap([[1 1 1];map]);
axis equal
axis off
colorbar
% print -dpng conductivity.png