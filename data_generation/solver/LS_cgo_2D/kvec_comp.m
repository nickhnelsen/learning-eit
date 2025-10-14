clear variables;

% Collection of complex generalized nonzero wave numbers k 
R       = 13; % g1.m cannot handle R=14 and higher
N       = 64;
h       = R/(N-1);
K       = h/2 + [0:N-1]*h;
K       = [-fliplr(K),K];
[K1,K2] = meshgrid(K);
kvec    = K1+1i*K2;
kvec    = kvec(abs(kvec)<R);

% Save the result to file
mkdir('.','data')
save data/kvec kvec R N h K1 K2
