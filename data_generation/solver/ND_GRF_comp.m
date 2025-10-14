% Nicholas H. Nelsen Jan. 2022

% Reproducibility
rng(2022,'twister');

% Load mesh and decomposed geometry matrix
% (precomputed with the routine mesh_comp.m)
load data/mesh p e t dgm

% NOTE: Order of truncated trigonometric basis (power of two)
Ntrig = 16;

% NOTE: Number of solves to probe the action of the NtD map
Nsolves = 128;

% NOTE: Regression Setup
gam2 = 1e-9*1;      % Actual variance of white noise polluting PDE solutions
nug = 1e-9*1;       % Penalty parameter modeling variance of white noise polluting PDE solutions
prior_std = 1e1;    % Stdev of prior on NtD map

% NOTE: Boundary GRF setup
K_bd = 1024;    % Resolution of GRF input data on the boundary (power of two)
tau = 5;
alpha = 1.1;    % alpha >= 0, smaller gives better convergence in HS norm (1.1 empirically best)

% We use the fact that in mesh_comp.m the unit circle was 
% divided into the following four segments: 
% (1) [pi,3*pi/2], (2) [3*pi/2,0], (3) [0,pi/2], (4) [pi/2,pi]
fii = pi + (e(5,:).'-1)*pi/2 + e(3,:).'*pi/2;           % takes value in [pi, 3pi]
% Sort the angles and arrange the corresponding values accordingly
[fii,ind] = sort(fii);
% Here we assume that the angles fii are equidistant
Dfii = fii(2)-fii(1);

% Angles for Neumann data
torus = (0:2*pi/K_bd:2*pi).';
torus = torus(1:end-1);
theta = wrapTo2Pi(fii);

% Allocate output arrays
neumann_array = zeros(Nsolves,K_bd);
dirichlet_array = zeros(Nsolves,K_bd);
    
% Build NtoD matrix via linear regression in Fourier space
Nvec  = [(-Ntrig:-1),(1:Ntrig)];
K_trig = length(Nvec);
Ygt = zeros(Nsolves,K_trig);
X = zeros(Nsolves,K_trig);
for nnn = 1:Nsolves
    % Sample GRF
    grf = GRFper(K_bd, tau, alpha);
    neumann_array(nnn,:) = grf;
    xtemp = interp1(torus,grf,theta,'spline');
    X(nnn,:) = 1/sqrt(2*pi)*Dfii*exp(-1i*Nvec.'*theta.')*xtemp;
    
    % Solve elliptic PDE with FEM
    bc_func = @(pvar,evar,~,~)BoundaryDataGRF(grf,pvar,evar);
    u = assempde(bc_func,p,e,t,'FEMconductivity',0,0);
    
%     figure(1)
%     clf
%     pdesurf(p,t,real(u))
%     drawnow
%     pause
    
    % Compute trace of solution (expressed in the finite element basis)
    u_tr = u(e(1,:));
    u_tr = u_tr(ind);
    u_tr = u_tr - mean(u_tr);   % subtract mean for uniqueness
    u_d = interp1(theta,u_tr,torus,'spline');
    dirichlet_array(nnn,:) = u_d - mean(u_d);
    
    % Project the traces into truncated trigonometric basis. 
    Ygt(nnn,:) = 1/sqrt(2*pi)*Dfii*exp(-1i*Nvec.'*fii.')*u_tr;
    disp(['Done ', num2str(nnn), ' out of ', num2str(Nsolves)])
end
noise = sqrt(gam2)*randn(size(Ygt));
Y = Ygt + noise;
XX = X'*X/Nsolves;
XY = X'*Y/Nsolves;
if nug < 1e-13
%     NtoD = (pinv(XX)*XY).';
    NtoD = lsqminnorm(X,Y).';
else
%     NtoD = ((XX + nug*eye(size(XX)))\XY).';   % Tikhonov regularization
    NtoD = zeros(size(XX));
    prior = prior_std*abs(1./Nvec);     % Scaled exact eigenvalues of NtD map for unit conductivity
    XXdiag = diag(XX);
    for j = 1:K_trig
        XXreg = XX;
        XXreg(j,j) = XXdiag(j) + nug/(Nsolves*prior(j)^2);
        NtoD(j,:) = XXreg\XY(:,j);
    end
end
NtoD = (NtoD + NtoD')/2;  % self-adjoint part


% TODO: Also try to learn unbounded DN directly from noisy X,Y pairs
% YY = Y'*Y/Nsolves;
% YX = Y'*X/Nsolves;
% if nug < 1e-13
%     DN = lsqminnorm(Y,X).';
% else
%     DN = zeros(size(YY));
%     prior = prior_std*abs(Nvec);       % Scaled exact eigenvalues of DN map for unit conductivity
%     YYdiag = diag(YY);
%     for j = 1:K_trig
%         YYreg = YY;
%         YYreg(j,j) = YYdiag(j) + nug/(Nsolves*prior(j)^2);
%         DN(j,:) = YYreg\YX(:,j);
%     end
% end
% DN = (DN + DN')/2;  % self-adjoint part
% DN = [DN(:,1:Ntrig),zeros(2*Ntrig,1),DN(:,Ntrig+1:end)];
% DN = [DN(1:Ntrig,:);zeros(1,2*Ntrig+1);DN(Ntrig+1:end,:)];


% Save result to file
save data/ND NtoD Nvec Ntrig neumann_array dirichlet_array nug gam2 prior_std tau alpha