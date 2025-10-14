% Pre-compute Fadeev Green's function because MATLAB expint function is slow!
clear variables;

% Folder to save data
data_folder = "/media/nnelsen/SharedHDD2TB/datasets/eit/";

% Size of the computational grid is 2^M times 2^M. Larger values of |k|
% typically need only a coarser grid. Values of M less than 7 probably will
% not give reasonable results.
M = 8;

% The potential is supported in a disc D(0,Rpot) with Rpot defined here
Rpot = 1;

% Size of the square giving the periodic structure on the plane.
% If the potential is supported in a disc D(0,Rpot), then s must be chosen so
% that s > 2*Rpot. In the case of the potential being support in the unit
% disc, a good choice is s=2.1.
s = 2.1;

% Batch size for Green's pre-comp
bsize = 1;

% Get the grid points and parameters for only one grid using the two-grid function GV_grids. 
[x1,x2,h,~,~,~] = GV_grids(M, M+1, s);
z = x1+1i*x2; % shape (2^M, 2^M)

% Pre-compute Green's function
load data/kvec kvec R N
kvec = reshape(kvec, [], 1);
K = length(kvec);
fund = zeros([size(x1), K]);

% Avoid singularity at origin by setting value there to zero
idx = 2^(M-1)+1; 
z0 = z;
z0(idx,idx) = 1/2; % any nonzero value

% TODO: batch over z0 too (batch over the flattened product batch_mul(z0,kvec) to avoid g1.m slowdown
% Loop to save time and memory
c = 0;
tic;
while 1
    if K - c <= 0
        break
    elseif K - c >= bsize
        batch = bsize;
    else
        batch = K - c;
    end
    
    kbatch = kvec(c+1:c+batch);
    kbatch = reshape(kbatch, [], 1);
    
    % Compute
    fund(:, :, c+1:c+batch) = green_faddeev(z0, kbatch); % shape (N, N, batch)

    % Update
    c = c + batch;

    % Monitor the run
    disp(['Done ', num2str(c), ' out of ', num2str(K), ' Greens functions '])
end
toc;

% Adjust for origin
fund(idx,idx,:) = 0;

% Save the result to file
save_path = append(data_folder, "green_LS_R" + string(R) + "_N" + string(N));
save(save_path,'fund','M','Rpot','s','x1','x2',...
    'h','z','-v7.3','-nocompression')
