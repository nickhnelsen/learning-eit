% Compute the Neumann-to-Dirichlet map of the conductivity
%
% Samuli Siltanen June 2012; Nicholas H. Nelsen Jan. 2022

% Load mesh and decomposed geometry matrix
% (precomputed with the routine mesh_comp.m)
load data/mesh p e t dgm

% Order of trigonometric approximation
% NOTE: change resolution/accuracy of DtN approximation here (power of two)
Ntrig = 16;

% We use the fact that in mesh_comp.m the unit circle was 
% divided into the following four segments: 
% (1) [pi,3*pi/2], (2) [3*pi/2,0], (3) [0,pi/2], (4) [pi/2,pi]
fii = pi + (e(5,:).'-1)*pi/2 + e(3,:).'*pi/2;           % takes value in [pi, 3pi], correct?
% fii = pi*0 + (e(5,:).'-1)*pi/2 + e(3,:).'*pi/2;
% Sort the angles and arrange the corresponding values accordingly
[fii,ind] = sort(fii);
% Here we assume that the angles fii are equidistant
Dfii = fii(2)-fii(1);
    
% Build NtoD matrix element by element
Nvec  = [(-Ntrig : -1),(1 : Ntrig)];
nnv = length(Nvec);
NtoD  = zeros(nnv);
for nnn = 1:nnv
    % Power of trigonometric basis function used as boundary data. 
    % We save n to disc, and it will be loaded by function BoundaryData.m.
    n = Nvec(nnn); 
    save data/BoundaryDataN n
        
    % Solve elliptic PDE with FEM
    u = assempde('BoundaryData',p,e,t,'FEMconductivity',0,0);
    
%     figure(1)
%     clf
%     pdesurf(p,t,real(u))
%     drawnow
%     pause
    
    % Compute trace of solution
    u_tr = u(e(1,:));
    u_tr = u_tr(ind);
    u_tr = u_tr - mean(u_tr);   % subtract mean for uniqueness
    
    % Project the traces into truncated trigonometric basis. 
    NtoD(:,nnn) = 1/sqrt(2*pi)*Dfii*exp(-1i*Nvec.'*fii.')*u_tr;
    disp(['Done ', num2str(nnn), ' out of ', num2str(nnv)])
end
NtoD = (NtoD + NtoD')/2;  % self-adjoint part

% Save result to file
save data/ND NtoD Nvec Ntrig