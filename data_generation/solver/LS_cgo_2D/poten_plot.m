clear variables;

% Plot the conductivity type potential defined in file poten.m,
% related to the conductivity defined in file sigma.m

N = 512;

% Set font size
fsize = 14;

% Create evaluation points
t       = linspace(-1,1,N);
[x1,x2] = meshgrid(t);
z       = x1 + 1i*x2;

% Evaluate
c = poten(z);
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

% ///////// 1D below /////////
% % Plot parameters
% lwidth  = 2;
% fsize   = 10;
% fsize2D = 10;
% 
% % Create plot window
% figure(1)
% clf;
% 
% % Create plot points for 1D plot
% t = linspace(0,1,1000);
% 
% % Evaluate potential
% poten1D = poten(t);
% 
% % Plot the potential
% subplot(1,2,2)
% plot(t,poten1D,'r','linewidth',lwidth)
% set(gca,'xtick',[0 .5 1],'fontsize',fsize)
% ytickvec = [-20:10:30];
% set(gca,'ytick',ytickvec,'fontsize',fsize)
% axis([0 1 -20 30])
% box off
% axis square
% xlabel('|z|','fontsize',fsize)
% title('Profile of potential q','fontsize',fsize)
% 
% % Create plot points for mesh plot
% x       = linspace(-1,1,200);
% [x1,x2] = meshgrid(x);
% 
% % Evaluate potential
% poten2D = poten(abs(x1+1i*x2));
% poten2D(abs(x1+1i*x2)>1) = NaN;
% 
% % Plot potential as mesh plot
% subplot(1,2,1)
% surf(x1,x2,poten2D)
% axis([-1 1 -1 1 -20 30])
% ax = gca;
% set(ax,'ztick',[-20:10:30],'fontsize',fsize2D)
% set(ax,'xtick',[-1 0 1],'fontsize',fsize2D)
% set(ax,'ytick',[-1 0 1],'fontsize',fsize2D)
% axis square
% colormap(jet)
% shading interp
% title('Potential','fontsize',fsize)
% xlabel('x','fontsize',fsize)
% ylabel('y','fontsize',fsize)
% 
% % Save picture to file
% %print -dpng poten.png


