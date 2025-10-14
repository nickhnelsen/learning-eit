% Written by:
% Nicholas H. Nelsen
% California Institute of Technology
% Email: nnelsen@caltech.edu

% Forward map run

% Last updated: Jan. 2022

clc; clear variables; close all;

%% Setup
GRF_FLAG = true;

%% Input a single random conductivity and plot it
% set_conductivity_comp;
conductivity_plot;

%% Obtain NtD map
tic;
if GRF_FLAG
    ND_GRF_comp;        % Random GRF Neumann data
else
    ND_comp;            % Fourier Neumann data
end
toc;

figure(100)

subplot(1,2,1)
imagesc(real(NtoD))
axis equal
axis off
colormap parula

subplot(1,2,2)
imagesc(imag(NtoD))
axis equal
axis off
colormap parula

%% Obtain DtN map from NtD map
if GRF_FLAG
    ex2DN_GRF_comp;
else
    ex2DN_comp;
end

figure(101)

subplot(1,2,1)
imagesc(real(DN))
axis equal
axis off
colormap parula

subplot(1,2,2)
imagesc(imag(DN))
axis equal
axis off
colormap parula

%% Error between exact DtN and NtD maps vs noisy learned maps

% TODO: also check Bochner norms for DN and ND? What smoothness parameter for each norm?
if GRF_FLAG
    disp(' ');

    load data/ex_f_maps fDN fNtoD
    load data/ND NtoD Ntrig
    load data/ex2DN DN 

    NvecDN = (-Ntrig:Ntrig).';
    NvecNtoD = ([-Ntrig:-1,1:Ntrig]).';
    facDN = ((1 + abs(NvecDN).^2).^(-1/4))*((1 + abs(NvecDN.').^2).^(-1/4));
    facNtoD = ((0 + abs(NvecNtoD).^2).^(1/4))*((0 + abs(NvecNtoD.').^2).^(1/4));

    disp('Full');
    norm(facDN.*(fDN - DN))/norm(facDN.*(fDN))          % H^1/2 to H^-1/2
    norm(facDN.*(fDN - DN),'fro')/norm(facDN.*(fDN),'fro')          % HS(H^1/2;H^-1/2)
    norm(facNtoD.*(fNtoD - NtoD))/norm(facNtoD.*fNtoD)  % .H^-1/2 to .H^1/2
    norm(facNtoD.*(fNtoD - NtoD),'fro')/norm(facNtoD.*fNtoD,'fro')  % HS(.H^-1/2;.H^1/2)
    norm(fNtoD - NtoD)/norm(fNtoD)                      % L^2 to L^2
    norm(fNtoD - NtoD,'fro')/norm(fNtoD,'fro')          % HS(L^2, L^2)

    disp('Real');
    norm(facDN.*real(fDN - DN))/norm(facDN.*real(fDN))              % H^1/2 to H^-1/2
    norm(facDN.*real(fDN - DN),'fro')/norm(facDN.*real(fDN),'fro')  % HS(H^1/2 to H^-1/2)
    norm(facNtoD.*real(fNtoD - NtoD))/norm(facNtoD.*real(fNtoD))    % .H^-1/2 to .H^1/2
    norm(facNtoD.*real(fNtoD - NtoD),'fro')/norm(facNtoD.*real(fNtoD),'fro') % HS(.H^-1/2 to .H^1/2)
    norm(real(fNtoD - NtoD))/norm(real(fNtoD))
    norm(real(fNtoD - NtoD),'fro')/norm(real(fNtoD),'fro')

    disp('Imag');
    norm(facDN.*imag(fDN - DN))/norm(facDN.*imag(fDN))              % H^1/2 to H^-1/2
    norm(facDN.*imag(fDN - DN),'fro')/norm(facDN.*imag(fDN),'fro')  % HS(H^1/2 to H^-1/2)
    norm(facNtoD.*imag(fNtoD - NtoD))/norm(facNtoD.*imag(fNtoD))    % .H^-1/2 to .H^1/2
    norm(facNtoD.*imag(fNtoD - NtoD),'fro')/norm(facNtoD.*imag(fNtoD),'fro') % HS(.H^-1/2 to .H^1/2)
    norm(imag(fNtoD - NtoD))/norm(imag(fNtoD))
    norm(imag(fNtoD - NtoD),'fro')/norm(imag(fNtoD),'fro')
end