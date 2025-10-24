% Requires function files: GRFcos.m, GRFper.m, BoundaryDataGRF.m,
%       get_cond_data.m, conductivityGRF.m, FEMconductivityGRF.m
%
% Script to generate large Fourier basis truth set of ND maps
% Nicholas H. Nelsen Mar. 2022

% clc; clear variables; close all;
function []=data_generation_fND_script(seed,N_cond,N_solves,...
    tau_m,tau_p,al_m,al_p,rho_m,rho_p,cr_m,cr_p)

addpath('./solver/');
warning('off','MATLAB:nearlySingularMatrix');

% Process strings
tau_m = str2double(tau_m);
tau_p = str2double(tau_p);
al_m = str2double(al_m);
al_p = str2double(al_p);
rho_m = str2double(rho_m);
rho_p = str2double(rho_p);
cr_m = str2double(cr_m);
cr_p = str2double(cr_p);

% NOTE: Level set model for conductivity setup
K_cond = 256;                   % Sample GRF resolution
N_cond = str2num(N_cond);       % (10) Number of conductivity/DtN pairs
tau_cond = [tau_m;tau_p];       % min 10 max 25
alpha_cond = [al_m;al_p];       % min 4 max 5.5
rho = [rho_m;rho_p];            % Outer truncation radius to conductivity value 1; min .65 max .9
contrast_ratio = [cr_m;cr_p];   % Max divided by min conductivity; min 10^1 max 10^3 

% NOTE: (512) Number of solves to probe the action of the NtD map
N_solves = str2num(N_solves);         

% NOTE: Reproducibility
seed = str2num(seed);   % (202203);

% NOTE: Dataset path
SAVE_AFTER = 3;       % after number of OUTER loops
if N_cond <= SAVE_AFTER
    SAVE_AFTER = 1;
end
str0 = "S" + num2str(seed) + "_";
str1 = "No" + num2str(N_cond) + "_";
str2 = "Ni" + num2str(N_solves) + "_";
str3 = "Ro" + num2str(K_cond);
save_path = append("./eit_bin_fND_",str0,str1,str2,str3);

% Derived
rng(seed,'twister');
load ./solver/data/mesh p e t      % Load mesh (precomputed in mesh_comp.m)

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
rho_c = rho_m+(rho_p-rho_m).*rand(N_cond,1);
cr_c = 10.^(log10(cr_m)+(log10(cr_p)-log10(cr_m)).*rand(N_cond,1));
for oo = 1:N_cond
    s2=tic;
    % Get raw data and gridded interpolant
    [raw_cond, interp_cond] = get_cond_data(K_cond,tau_c(oo),al_c(oo),rho_c(oo),cr_c(oo));
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
        save(save_path,'kvec_array','ntd_array','cond_array','tau_cond','alpha_cond','rho',...
    'contrast_ratio','seed','-v7.3','-nocompression')
    end
    disp(['(Seed ', num2str(seed), ') Done ', num2str(oo), ' out of ', num2str(N_cond), ' outer loops',...
        ' in ', num2str(toc(s2)/60), ' minutes'])
end
% ntd_array = (ntd_array + permute(conj(ntd_array),[1 3 2]))/2;
save(save_path,'kvec_array','ntd_array','cond_array','tau_cond','alpha_cond','rho',...
    'contrast_ratio','seed','-v7.3','-nocompression')
disp(['Done (full data generation) ', 'in ', num2str(toc(s1)/3600), ' hours'])
rmpath('./solver/');

end
