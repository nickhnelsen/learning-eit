% Implements the real-linear operator
%
% w |->  w-(1/(pi k))*(TR.conj(w)),
%
% where ``*'' means convolution and ``.'' means pointwise multiplication.
%
% Samuli Siltanen June 2012

function result = DB_oper(w, fundfft, TR, ~, ~, M, h, ~, ~, Rind, Nind)


% Reshape w to square 
N            = 2^M;
result       = zeros(N, N);
result(Rind) = w(1:Nind) + 1i*w((Nind+1):end);

% Apply real-linear operator
result = result - h^2*ifft2(fundfft .* fft2(TR.*conj(result)));

% Construct result as a vector with real and imaginary parts separate
result = [real(result(Rind));imag(result(Rind))];













