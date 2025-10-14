% Written by:
% Nicholas H. Nelsen
% California Institute of Technology
% Email: nnelsen@caltech.edu

% Return a sample of a level set thresholded Gaussian random field on
% [-1,1]^2 with
%       mean function m = 0
%       covariance operator C = (tau/2)^(2*alpha-2)*(-Delta + (tau/2)^2*I)^(-alpha)
% restricted to the unit disk

% Last updated: Jan. 2022

function gamma = conductivityGRF(grf_interp, z)
% Input
%   grf_interp: gridded_interpolant object
%   z: (N,N) or (1,N^2) complex matrix of planar evaluation points, given as complex numbers.
% Output
%   gamma: (N,N) or (1,N^2) real-valued conductivity on [-1,1]^2 

    % Initialize
    [zrow,zcol] = size(z);
    z = z(:);
    x1 = real(z);
    x2 = imag(z);
    
    % Evaluate conductivity
    gamma = grf_interp(x1,x2);
    
    % Reshape to original shape
    gamma = reshape(gamma,[zrow,zcol]);
end