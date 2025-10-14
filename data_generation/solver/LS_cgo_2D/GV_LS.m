% This routine is needed for G Vainikko's method of solution for the 
% Lippmann-Schwinger equation. Implements the operator [I + g* (V. )]w, 
% where * denotes convolution and . denotes pointwise multiplication.
%
% Samuli Siltanen March 2012

function result = GV_LS(w, fundfft, V, h)
% w,        (N^2*K, 1)
% fundfft,  (N, N, K)
% V,        (N, N)
% h,        (scalar)

% Reshape w to flattened vector if not already (reshape(w,[],1) is faster than w(:))
w = reshape(w, [], 1);

% Temporary variable for w given on the square grid
result = reshape(w, size(fundfft, 1), size(fundfft, 2), []); % shape (N, N, K)

% Convolution-multiplication operator (fundfft is always in fft2 size)
result = h^2*ifft2(fundfft .* fft2(V.*result));

% Final result needs the identity operator
result = w + reshape(result, [], 1);


