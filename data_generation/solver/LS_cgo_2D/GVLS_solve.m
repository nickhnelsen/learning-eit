% One-grid solver of the Lippmann-Schwinger equation
%
% (LS)  w = 1 - g * (V w), 
%
% where g is a Green function. Equation (LS) appears in scattering problems for the Helmholtz
% equation and in the computation of exponentially growing solutions of Faddeev to the Schrodinger 
% equation. 
%
% To use this solver it is needed to write Matlab routines for the Green function g and the 
% multiplicator function V. Names of these routines are given as string arguments. 
% Both functions should take as argument a matrix of complex evaluation points.
% It is also possible to pass one extra parameter as argument to these routines.
%
% This solver is based on GMRES. See Gennadi Vainikko 1997: Fast solvers of the Lippmann-Schwinger 
% equation, in "Direct and inverse problems of mathematical physics", Newark, DE,
% Int.Soc.Anal.Appl.Comput. 5, pp. 423-440, Kluwer acad.publ., Dordrecht 2000
%
% Arguments:
% gname      (func handle) name of the Green function 
% gparam     parameter passed as argument to the Green function (use [] for no parameter)
% mname      (func handle) name of the multiplicator function V 
% M          solution is given on a grid of size [2^M, 2^M] = [N, N]
% s          (positive number) gives the square [-s,s-h]^2 filled by the grid where solution is given
% R          (positive number) Multiplicator function V is supported in D(0,R)
%
% Returns:
% w          (complex) solution of the Lippmann-Schwinger equation
% x1         x1-coordinates of the grid
% x2         x2-coordinates of the grid (example: mesh(x1,x2,real(w)) )
% h          grid parameter
%
% Calls to: GV_grids, GV_LS, gname, mname, gmres, fft2
%
% Samuli Siltanen April 2012 and vectorized by Nicholas Nelsen 2023

function w = GVLS_solve(gprecomp, gparam_idx, mprecomp, M, R, x1, x2, h, init)

% Construct "incident wave" function rhs=1 as vertical vector for RHS
K = length(gparam_idx);             % gparam = idx has shape (K,1)
rhs = ones([size(x1),K]);
rhs = reshape(rhs,[],1);            % shape (N^2*K, 1)

% Evaluate Green's function.
% Avoid singularity at origin by setting value there to zero
c               = 2^(M-1)+1; 
z               = (x1+1i*x2);       % shape (N, N)
z(c,c)          = 1;
fundfft         = gprecomp(:, :, gparam_idx); % shape (N, N, K)
fundfft(c,c,:)  = 0;

% Smooth truncation of Green's function near the boundary
ss              = abs(min(min(x1)));
temp            = abs(z)>=ss;                   % big indices
temp            = repmat(temp,1,1,K);
fundfft(temp)   = 0;
w               = (abs(z)<ss) & (abs(z)>2*R);   % medium indices
temp            = (1-(abs(z(w))-2*R)/(ss-2*R)); % epsilon = ss-2*R
w               = repmat(w,1,1,K);
temp            = reshape(fundfft(w),[],K).*temp;
fundfft(w)      = reshape(temp, [], 1);

% Compute discrete Fourier transform of Green's function
fundfft = ifftshift(fundfft);
fundfft = fft2(fundfft);

% Solve Lippmann-Schwinger equation using gmres, output shape (N^2*K,1)
w = gmres('GV_LS', rhs, 20, 1e-8, 25, [], [], init, fundfft, mprecomp, h);

% Reshape w  to original shape (N, N, K)
w = reshape(w, 2^M, 2^M, []);

end


