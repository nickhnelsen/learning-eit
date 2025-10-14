% Plot the results of ex2tBIErecon_comp.m
%
% Samuli Siltanen June 2012; Nicholas H. Nelsen Dec. 2021

% Multiple of upsampling the reconstruction (see below)
USmultiple = 8;

% Load precomputed reconstruction and its evaluation points
load data/ex2recon x1 x2 recon
recon = reshape(real(recon),size(x1));

% We want to show the original conductivity for comparison at higher
% resolution. For this we construct a finer grid for the square [-1,1]^2
t = linspace(-1,1,USmultiple*size(x1,1));
[X1,X2] = meshgrid(t);

% Evaluate the original conductivity on the finer grid
orig = conductivity(X1+1i*X2);

% Record combined minimum and maximum of the reconstruction and original
MIN = min(min(orig(:)),min(recon(:)));
MAX = max(max(orig(:)),max(recon(:)));

% Create customized colormap for white background 
colormap parula
MAP = colormap;
M = size(MAP,1); % Number of rows in the colormap
bckgrnd = [1 1 1]; % Pure white color
MAP = [bckgrnd;MAP];

% Find index vectors for points inside the unit disc
discind = (abs(x1+1i*x2)<1);
Discind = (abs(X1+1i*X2)<1);

% Modify the functions for constructing white background
cstep = (MAX-MIN)/(M-1); % Step size in the colorscale from min to max
recon(~discind) = MIN-cstep;

orig(~Discind) = MIN-cstep;

% Upsample the reconstruction to the same matrix size than orig
[row,col] = size(recon);
recon = recon(:).';
recon = repmat(recon,USmultiple,1);
recon = recon(:);
recon = reshape(recon,USmultiple*row,col);

recon = recon.';
[row,col] = size(recon);
recon = recon(:).';
recon = repmat(recon,USmultiple,1);
recon = recon(:);
recon = reshape(recon,USmultiple*row,col);

recon = recon.';


% Plot truth and reconstruction from DtN map
try
    close(1)
catch
end
% figure(2)
figure(22)
clf
imagesc([orig,recon])
axis equal
axis off
colormap(MAP)

% Write image to file
% print -dpng ex2recon.png