% Inverse diagonal preconditioner (Monte Carlo)

function result = diag_LS(w, fundfft, V, h, nsamp)
% w,            (N^2*K, 1)
% fundfft,      (N, N, K)
% V,            (N, N)
% h,            (scalar)
% nsamp,        (scalar)

% Reshape w to flattened vector if not already (reshape(w,[],1) is faster than w(:))
w = reshape(w, [], 1);

% Rademachers
Om = -3 + 2*randi(2,[size(fundfft),nsamp]); % Random signs of shape (N, N, K, nsmap)

% Convolution-multiplication operator (fundfft is always in fft2 size)
result = Om + h^2*ifft2(fundfft .* fft2(V.*Om));

% Form diagonal estimator
result = reshape(Om.*result,nsamp,[]);
result = sum(result,1)/nsamp;
result = w./reshape(result,[],1); % invert diagonal matrix
end