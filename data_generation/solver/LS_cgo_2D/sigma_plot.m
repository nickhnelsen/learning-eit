clear variables;

% Plot the conductivity described in the file sigma.m
N = 1024;

% Set font size
fsize = 14;

% Create evaluation points
t       = linspace(-1,1,N);
[x1,x2] = meshgrid(t);
z       = x1 + 1i*x2;

% Evaluate conductivity
c = sigma(z);
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

% /////////////// 1D below ///////////////
% % Plot parameters
% lwidth  = 2;
% fsize   = 10;
% fsize2D = 10;
% 
% % Create plot window
% figure(1)
% clf
% 
% % Create plot points for 1D plot
% t = linspace(0,1,1000);
% 
% % Evaluate conductivity
% sigma1D = sigma(t);
% 
% % Plot the conductivity
% subplot(1,2,2)
% plot(t,sigma1D,'r','linewidth',lwidth)
% set(gca,'xtick',[0 .5 1],'fontsize',fsize)
% set(gca,'ytick',[0:.5:2.5],'fontsize',fsize)
% axis([0 1 1 2.5])
% box off
% axis square
% xlabel('|z|','fontsize',fsize)
% title('Profile of conductivity','fontsize',fsize)
% 
% % Create plot points for mesh plot
% x       = linspace(-1,1,200);
% [x1,x2] = meshgrid(x);
% 
% % Evaluate conductivity
% sigma2D = sigma(abs(x1+1i*x2));
% sigma2D(abs(x1+1i*x2)>1) = NaN;
% 
% % Plot conductivity as mesh plot
% subplot(1,2,1)
% surf(x1,x2,sigma2D)
% axis([-1 1 -1 1 1 2.5])
% ax = gca;
% set(ax,'ztick',[0:.5:2.5],'fontsize',fsize2D)
% set(ax,'xtick',[-1 0 1],'fontsize',fsize2D)
% set(ax,'ytick',[-1 0 1],'fontsize',fsize2D)
% axis square
% colormap(jet)
% shading interp
% title('Conductivity','fontsize',fsize)
% xlabel('x','fontsize',fsize)
% ylabel('y','fontsize',fsize)
% 
% % Save picture to file
% %print -dpng sigma.png


