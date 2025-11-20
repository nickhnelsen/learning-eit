% Construct grid in the k-plane for evaluation of the scattering transform
%
% Samuli Siltanen June 2012

% Collection of complex generalized nonzero wave numbers k 
R       = 9;
h       = 2*R/126; % choose this so that size(tt) = 128 points
N       = round(R/h);
K       = h/2 + [0:N]*h;
K       = [-fliplr(K),K];
tt      = K;
[K1,K2] = meshgrid(K);
Kvec    = K1+1i*K2;
Kvec    = Kvec(abs(Kvec)<R);

% Save result to file
tMAX = R;
% save data/ex2Kvec Kvec R h K1 K2 tt tMAX
save('/media/nnelsen/SharedHDD2TB/datasets/eit/dbar/data/ex2Kvec.mat', 'Kvec', 'R', 'h', 'K1', 'K2', 'tt', 'tMAX');

