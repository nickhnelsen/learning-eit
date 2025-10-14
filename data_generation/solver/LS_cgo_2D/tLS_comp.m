% This routine is for computing the scattering transform at the k-grid
% created by routine kvec_comp.m.
%
% We use the Lippmann-Schwinger-type equation approach for computing t(k).
% This involves knowing the conductivity, not starting from boundary 
% measurements as in the EIT problem.
%
% Samuli Siltanen April 2012 and Nicholas Nelsen Jan 2023


% Data folder
data_folder = "/media/nnelsen/SharedHDD2TB/datasets/eit/";

% Batch size for k-values
bsize = 1;

% Load the precomputed quantities
load data/kvec kvec R N
load(append(data_folder,"green_LS" + "_R" + string(R) + "_N" + string(N)), ...
            'fund', 'M', 'Rpot', 'x1', 'x2', 'h', 'z')

% Initialize the result
sz_kvec = size(kvec);
len_kvec = length(kvec(:));
tLS = zeros(len_kvec,1);

% Pre-compute the spatial unit disk
mz = poten(z);
zind = abs(z)<=Rpot;        % shape (2^M, 2^M)
z = z(zind);                % shape (J)
pz = poten(z);              % shape (J)

% Initial guess
Nmu = ones([size(x1),bsize]);
Nmu = reshape(Nmu,[],1);    % shape (N^2*K, 1)

% Loop over batches of k-values
tic;
c = 0;
while 1
    if len_kvec - c <= 0
        break
    elseif len_kvec - c >= bsize
        batch = bsize;
    else
        batch = len_kvec - c;
        Nmu = Nmu(:,:,end-batch+1:end);
    end
    
    idx_batch = (c+1:c+batch).';
    kbatch = kvec(c+1:c+batch);
    kbatch = reshape(kbatch, [], 1);
    
    % Compute complex geometric optics solution using the
    % Lippmann-Schwinger type equation
    Nmu = GVLS_solve(fund, idx_batch, mz, M, Rpot, x1, x2, h, reshape(Nmu,[],1));
    
    % Integrate
    pNmu = reshape(Nmu(repmat(zind,1,1,batch)),[],batch);
    pNmu = pz.*pNmu;  % shape (J, K), q times mu, batched multiplied 
    zk = batch_mul(z,kbatch);    % shape (J, K), k times z
    tLS(c+1:c+batch) = (h^2)*trapz(exp(1i*(zk+conj(zk))).*pNmu); % shape (1, K)

    % Update
    c = c + batch;
    
    % Monitor the run
    disp(['Done ', num2str(c), ' out of ', num2str(len_kvec)])
end
toc;

% Save the result to file.
tLS = reshape(tLS, sz_kvec);
save data/tLS tLS kvec
