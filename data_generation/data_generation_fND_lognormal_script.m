% Requires files: GRFcos.m, BoundaryDataEXP.m,
%       get_cond_data_lognormaltrunc.m, conductivityGRF.m,
%       FEMconductivityGRF.m, mesh_nref4.mat, cutoff.m
%
% Script to generate large Fourier basis truth set of ND maps
% Lognormal conductivities
% Nicholas H. Nelsen Nov. 2025

function []=data_generation_fND_lognormal_script(seed,N_cond,N_solves,...
    tau_m,tau_p,al_m,al_p,rhom_m,rhom_p,rhop_m,rhop_p,scale_m,scale_p)

addpath('./solver/');
warning('off','MATLAB:nearlySingularMatrix');

% Process strings
tau_m = str2double(tau_m);
tau_p = str2double(tau_p);
al_m = str2double(al_m);
al_p = str2double(al_p);
rhom_m = str2double(rhom_m);
rhom_p = str2double(rhom_p);
rhop_m = str2double(rhop_m);
rhop_p = str2double(rhop_p);
scale_m = str2double(scale_m);
scale_p = str2double(scale_p);

% NOTE: Level set model for conductivity setup
K_cond = 256;                   % Sample GRF resolution
N_cond = str2num(N_cond);       % Number of conductivity/DtN pairs
tau_cond = [tau_m;tau_p];       % min 7 max 9
alpha_cond = [al_m;al_p];       % min 3 max 4
rhom = [rhom_m;rhom_p];         % Inner radius; min .5 max .55
rhop = [rhop_m;rhop_p];         % Outer truncation radius to conductivity value 1; min .85 max .95
scale = [scale_m;scale_p];      % cutoff scale; min 7.5 max 8.5

% NOTE: Number of solves to probe the action of the NtD map
N_solves = str2num(N_solves);         

% NOTE: Reproducibility
seed = str2num(seed);   % (202203);

% NOTE: Dataset path
SAVE_AFTER = 5;       % after number of OUTER loops
if N_cond <= SAVE_AFTER
    SAVE_AFTER = 1;
end
str0 = "S" + num2str(seed) + "_";
str1 = "No" + num2str(N_cond) + "_";
str2 = "Ni" + num2str(N_solves) + "_";
str3 = "Ro" + num2str(K_cond);
save_path = append("./eit_bin_fND_lognormal_",str0,str1,str2,str3);

% Derived
rng(seed,'twister');
load ./solver/data/mesh_nref4 p e t      % Load mesh (precomputed in mesh_comp.m)

% We use the fact that in mesh_comp.m the unit circle was 
% divided into the following four segments: 
% (1) [pi,3*pi/2], (2) [3*pi/2,0], (3) [0,pi/2], (4) [pi/2,pi]
fii = pi + (e(5,:).'-1)*pi/2 + e(3,:).'*pi/2;           % takes value in [pi, 3pi]
% Sort the angles and arrange the corresponding values accordingly
[fii,ind] = sort(fii);
dth = abs(fii(2)-fii(1));

% Angles for Dirichlet data
theta = wrapTo2Pi(fii);

% Allocate output arrays
cond_array = zeros(N_cond,K_cond,K_cond);
kvec_array = [1:N_solves/2,-N_solves/2+1:-1];
N_exp = length(kvec_array);
ntd_array = zeros(N_cond,N_exp,N_exp);

% Outer Loop: Sample conductivities
s1=tic;
% Sample hyperparameters
tau_c = tau_m+(tau_p-tau_m).*rand(N_cond,1);
al_c = al_m+(al_p-al_m).*rand(N_cond,1);
rhom_c = rhom_m+(rhom_p-rhom_m).*rand(N_cond,1);
rhop_c = rhop_m+(rhop_p-rhop_m).*rand(N_cond,1);
scale_c = scale_m+(scale_p-scale_m).*rand(N_cond,1);
for oo = 1:N_cond
    s2=tic;
    % Get raw data and gridded interpolant
    [raw_cond, interp_cond] = get_cond_data_lognormaltrunc(K_cond,tau_c(oo),al_c(oo),rhop_c(oo),rhom_c(oo),scale_c(oo));
    cond_func = @(pvar,tvar,~,~)FEMconductivityGRF(interp_cond,pvar,tvar);
    cond_array(oo,:,:) = raw_cond;
    
    % Inner Loop: Solve Neumann problems for Fourier modes
    for mm = 1:N_exp
        % Solve elliptic PDE with FEM
        bc_func = @(pvar,evar,~,~)BoundaryDataEXP(kvec_array(mm),pvar,evar);
        u = assempde(bc_func,p,e,t,cond_func,0,0);

        % Compute trace of solution (expressed in the finite element basis)
        u_tr = u(e(1,:));
        u_tr = u_tr(ind);
        u_tr = u_tr - mean(u_tr);   % subtract mean for uniqueness
        
        % Project into Fourier basis on [0,2\pi]_per
        ntd_array(oo,mm,:) = 1/sqrt(2*pi)*dth*exp(-1i*kvec_array.'*theta.')*u_tr;
    end
    
    % Save result to file after each SAVE_AFTER outer loops
    if mod(oo,SAVE_AFTER)==0
        save(save_path,'kvec_array','ntd_array','cond_array','tau_cond','alpha_cond','rhom',...
    'rhop','scale','seed','-v7.3','-nocompression')
    end
    disp(['(Seed ', num2str(seed), ') Done ', num2str(oo), ' out of ', num2str(N_cond), ' outer loops',...
        ' in ', num2str(toc(s2)/60), ' minutes'])
end
save(save_path,'kvec_array','ntd_array','cond_array','tau_cond','alpha_cond','rhom',...
    'rhop','scale','seed','-v7.3','-nocompression')
disp(['Done (full lognormal data generation) ', 'in ', num2str(toc(s1)/3600), ' hours'])
rmpath('./solver/');

end
