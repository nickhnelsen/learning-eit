function out = sigma(z)
% Requires set_sigma.m to be run at least once
load data/single_sigma sigma_interp
addpath ..;
out = conductivityGRF(sigma_interp, z);
rmpath ..;
end


% /////////// 1D below ////////////

% function out = sigma(z)
% rho    = 3/4;
% alpha  = 5;
% rr     = abs(z);
% ind    = (abs(rr)<(rho-1e-13));
% r      = rr(ind);
% F      = zeros(size(rr));
% F(ind) = (r.^2-rho^2).^4.*(1.5-cos(3*pi*r/(2*rho)));
% out    = (1+alpha*F).^2;

% % discontinuous with smoothing
% function out = sigma(z)
% zz = linspace(0,1,512);
% c1      = 1.5;
% c2      = 2;
% rho1    = 1/6;
% rho2    = 3/4;
% rr      = abs(zz);
% out     = ones(size(rr));
% ind     = (abs(rr)<(rho2-1e-13));
% out(ind) = c2;
% ind     = (abs(rr)<(rho1-1e-13));
% out(ind) = c1;
% 
% sig = 1e-2*2;
% gaussFilter = exp(-(abs(zz)-1/2).^ 2 / (2 * sig ^ 2));
% gaussFilter = gaussFilter / sum (gaussFilter); % normalize
% out = conv(out, gaussFilter, 'same');
% out(out < 1) = 1;
% ind = ind & (out < c1);
% out(ind) = c1;
% 
% out = interp1(abs(zz),out,abs(z),'spline');

