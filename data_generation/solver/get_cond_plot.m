% Plot the random conductivity 

% Choose conductivity parameters
N=1024;
tau=20;
alpha=4.5;
rho=0.75;
contrast_ratio=30;
sigma=25;

tau_ln=10;
alpha_ln=3.0;

tau_lnt=8;
alpha_lnt=3.6;
rhop_lnt=0.95;
rhom_lnt=0.5;
scale=8;

% Three phase
tau=15;
alpha=4.5;
rho=0.8;
val1=10;
val2=0.1;

% Set font size
fsize = 14;

% Create evaluation points
t       = linspace(-1,1,N);
[x1,x2] = meshgrid(t);
z       = x1 + 1i*x2;

% Evaluate conductivity
[c, ~] = get_cond_data_three_phase(N,tau,alpha,rho,val1,val2);
% [c, ~] = get_cond_data(N,tau,alpha,rho,contrast_ratio);
% [c, ~] = get_cond_data_smooth(N,tau,alpha,rho,contrast_ratio,sigma);
% [c, ~] = get_cond_data_lognormal(N,tau_ln,alpha_ln);
% [c, ~] = get_cond_data_lognormaltrunc(N,tau_lnt,alpha_lnt,rhop_lnt,rhom_lnt,scale);
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
% print -dpng conductivity_piecewise.png
% print -dpng conductivity_smooth.png
% print -dpng conductivity_lognormal.png
% print -dpng conductivity_lognormaltrunc.png