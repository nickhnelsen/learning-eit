% Approximate reconstruction of conductivity from truncated scattering data
% using the D-bar method.
%
% We optimize the computation by restricting the degrees of freedom to the
% values of the solution at grid points satisfying |k|<R.
%
% The routine ex2tBIE_comp.m must be computed before running this file.
%
% Samuli Siltanen June 2012; Nicholas H. Nelsen 2025

% Load precomputed scattering transform and its evaluation points
load('/media/nnelsen/SharedHDD2TB/datasets/eit/dbar/data/ex2Kvec.mat', 'Kvec', 'K1', 'K2', 'tt', 'tMAX');
% load data/ex2Kvec Kvec K1 K2 tt tMAX
load('/media/nnelsen/SharedHDD2TB/datasets/eit/dbar/data/ex2tBIE.mat', 'tBIE');
% load data/ex2tBIE tBIE
scatBIE = zeros(size(K1));
scatBIE(abs(K1+1i*K2)<tMAX) = tBIE;

% NOTE: Choose parameter M so gridsize=2^M for the computational grid
M = 6;
Mx = M; % NOTE: change Mx for higher resolution reconstruction
fudge = 2.3;
% fraction_tmax = 0.9;

% NOTE: Choose truncation radius R>0
R = 7; %fraction_tmax*tMAX; % TODO: around R=10 for clean data, R=7 or 8 for noisy data
if R>tMAX    error(['N_recon.m: Truncation radius R=', num2str(R), ' too big, must be less than ', num2str(tMAX)])
end

% Construct grid points
[k1,k2,h,~,~,~] = GV_grids(M, M+1, fudge*R);
k    = k1 + 1i*k2;
Rind = abs(k)<R;
Nind = round(sum(sum(double(Rind))));

% Evaluate scattering transform at the grid points using precomputed values
% and two-dimensional interpolation
k1vec   = k1(Rind);
k2vec   = k2(Rind);
scatvec = interp2(K1,K2,scatBIE,k1vec,k2vec,'spline');
scat = zeros(size(k1));
scat(Rind) = scatvec;

% Evaluate scattering transform divided by conj(k). Avoid singularity
% at the origin by setting value there to zero.
ktmp        = k;
ind0        = (abs(k)<1e-14); % Location of the origin in the grid
ktmp(ind0)  = 1;
scatk       = scat./conj(ktmp);
scatk(ind0) = 0;

% Evaluate Green's function 1/(pi*k). Avoid singularity at the origin by
% setting value there to zero.
fund       = 1./(pi*ktmp);
fund(ind0) = 0;

% Smooth truncation of Green's function near the boundary
s  = abs(min(min(k1)));
ep = s/10;
RR = (s-ep)/2;
bigind       = abs(k)>=s;
fund(bigind) = 0;
medind       = (abs(k)<s) & (abs(k)>2*RR);
fund(medind) = fund(medind).*(1-(abs(k(medind))-2*RR)/ep);

% Take FFT of the fundamental solution at this time
fundfft    = fft2(fftshift(fund));

% Construct reconstruction points
[x1,x2,~,~,~,~] = GV_grids(Mx, Mx+1, 1);
xvec = x1(:) + 1i*x2(:);
Nx    = length(xvec);

% Construct right hand side of the Dbar equation
rhs = [ones(Nind,1);zeros(Nind,1)];

% Initialize reconstruction
recon = ones(Nx,1);
iniguess = [ones(Nind,1);zeros(Nind,1)];

% Loop over points of reconstruction
% TODO: vectorize (may not be recommended due to GMRES sequential initial guess)
tic
for iii = 1:Nx
    % Current point of reconstruction
    x = xvec(iii);
    
    % Construct multiplicator function for the Dbar equation
    TR = 1/(4*pi)*scatk.*exp(-1i*(k*x+conj(k*x)));
    
    % Solve the real-linear D-bar equation with gmres keeping the real and
    % imaginary parts of the solution separate
    [w,~,~,~,~] = gmres('DB_oper', rhs, 50, 1e-5, 500, [], [], iniguess, fundfft, TR, k1, k2, M, h, k, R, Rind, Nind);
    
    % Use the current solution as the next initial guess
    iniguess = w;
    
    % Construct solution mu inside the unit disc
    mu = zeros(size(k1));
    mu(Rind) = w(1:Nind) + 1i*w((Nind+1):end);
    
    % Pick out the reconstructed conductivity value
    recon(iii) = (mu(ind0)).^2;
    
    % Monitor the run
    if mod(iii,20)==0
        disp(['Done (Dbar) ', num2str(iii), ' out of ', num2str(Nx)])
%         save data/ex2recon x1 x2 recon
        save('/media/nnelsen/SharedHDD2TB/datasets/eit/dbar/data/ex2recon.mat', 'x1', 'x2', 'recon');
    end
end
disp(['Done (Reconstruction) ', 'in ', num2str(toc), ' seconds.'])

% Write results to file
save('/media/nnelsen/SharedHDD2TB/datasets/eit/dbar/data/ex2recon.mat', 'x1', 'x2', 'recon');