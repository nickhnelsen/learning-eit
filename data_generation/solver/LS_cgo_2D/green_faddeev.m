% Faddeev Green's function g_k(x) 
%
% Arguments:
% x     complex evaluation points (nonzero)
% k     complex spectral parameter (nonzero)
%
% Returns:
% g     the Green's function g_k(x) = g_1(kx)
%
% Calls to:
% g1.m
%
% Samuli Siltanen March 2012 and Nicholas Nelsen Jan 2023

function g = green_faddeev(x,k)
g = g1(batch_mul(x,k));


