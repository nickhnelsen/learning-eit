% Plot reconstructions
clear variables; close all;

% Plot parameters
lwidth = 2;
fsize  = 16;
FLAG_SAVE = false;

% Load precomputed data
load data/recon x1 x2 recon

% Find index vector for points inside unit disc
z = x1+1i*x2;
discind = (abs(z)<1);

recon_dbar = ones(size(x1));
recon_dbar(discind) = recon;
recon_true = sigma(z);

% Plot truth only
c = recon_true;
c(abs(z)>1) = NaN;
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
title('True')

% Save image to file
if FLAG_SAVE
print -dpng recon_true_2D.png
end

% Plot recon only
c = recon_dbar;
c(abs(z)>1) = NaN;
figure(2)
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
title('LS reconstruction')

% Save image to file
if FLAG_SAVE
print -dpng recon_only_2D.png
end

% Plot pw error
c = abs(recon_dbar - recon_true);
disp(norm(c(discind))/norm(recon_true(discind)));
c(abs(z)>1) = NaN;
figure(3)
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
title('Pointwise error')

% Save image to file
if FLAG_SAVE
print -dpng recon_error_2D.png
end

% Plot side by side
% Create customized colormap for white background 
colormap parula
MAP = colormap;
M = size(MAP,1); % Number of rows in the colormap
bckgrnd = [1 1 1]; % Pure white color
MAP = [bckgrnd;MAP];

% Modify the function for constructing white background
MIN = min(min([recon_dbar,recon_true]));
MAX = max(max([recon_dbar,recon_true]));
cstep = (MAX-MIN)/(M-1); % Step size in the colorscale from min to max
recon_dbar(~discind) = MIN-cstep;
recon_true(~discind) = MIN-cstep;

% Plot dbar and true conductivities
figure(4)
clf
imagesc([recon_dbar,recon_true])
axis equal
axis off
colormap(MAP)
title('LS reconstruction (left) vs true (right)')

% Save image to file
if FLAG_SAVE
print -dpng recon_both_2D.png
end
