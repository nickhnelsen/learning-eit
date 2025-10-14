% Approximate reconstruction of conductivity from truncated scattering data
% using the D-bar method.
%
% We optimize the computation by restricting the degrees of freedom to the
% values of the solution at grid points satisfying |k|<R and |x|<1

% Load precomputed non-noisy scattering transform and its evaluation points
load data/kvec K1 K2 R
load data/tLS tLS
Rmax = R;
scat = zeros(size(K1));
scat(abs(K1+1i*K2)<Rmax) = tLS;

% Choose parameter M for the computational grid
M = 8;
Mx = M-1; % M or M+-1

% Choose truncation radius R>0
R = Rmax - 0.1;
if R>Rmax
    error(['Truncation radius R=', num2str(R), ' too big, must be less than ', num2str(Rmax)])
end
sfac = 2.1;

% Construct k-plane grid points
[k1,k2,h,~,~,~] = GV_grids(M, M+1, sfac*R);
k    = k1 + 1i*k2;
Rind = abs(k)<R;
Nind = round(sum(double(Rind), 'all'));

% Evaluate scattering transform at the grid points using precomputed values
% and two-dimensional interpolation
k1vec   = k1(Rind);
k2vec   = k2(Rind);
scatvec = interp2(K1,K2,scat,k1vec,k2vec,'spline');
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
fundfft = fft2(ifftshift(fund));

% Construct spatial domain grid points (unit disk)
[x1,x2,~,~,~,~] = GV_grids(Mx, Mx+1, 1);
xvec = x1 + 1i*x2;
xvec = xvec(abs(xvec)<1);
Nx   = length(xvec);

% Construct right hand side of the Dbar equation
rhs = [ones(Nind,1);zeros(Nind,1)];

% Initialize reconstruction
recon = ones(Nx,1);
iniguess = [ones(Nind,1);zeros(Nind,1)];

% Loop over points of reconstruction
tic;
for iii = 1:Nx    
    % Current point of reconstruction
    x = xvec(iii);
    
    % Construct multiplicator function for the Dbar equation
    TR = 1/(4*pi)*scatk.*exp(-1i*(k*x+conj(k*x)));
    
    % Solve the real-linear D-bar equation with gmres keeping the real and
    % imaginary parts of the solution separate
%     w = gmres('DB_oper', rhs, 50, 1e-6, 500, [], [], iniguess, fundfft, TR, k1, k2, M, h, k, R, Rind, Nind);
    w = gmres('DB_oper', rhs, 20, 1e-8, 25, [], [], iniguess, fundfft, TR, k1, k2, M, h, k, R, Rind, Nind);
    
    % Use the current solution as the next initial guess
    iniguess = w;
    
    % Construct solution mu inside the unit disc
    mu = zeros(size(k1));
    mu(Rind) = w(1:Nind) + 1i*w((Nind+1):end);
    
    % Pick out the reconstructed conductivity value
    recon(iii) = (real(mu(ind0)))^2;
    
    % Monitor the run
    if mod(iii,20)==0
        disp(['Done (Dbar) ', num2str(iii), ' out of ', num2str(Nx)])
    end
end
disp(['Done (Reconstruction) ', 'in ', num2str(toc), ' seconds.'])

% Save results to file
save data/recon x1 x2 recon