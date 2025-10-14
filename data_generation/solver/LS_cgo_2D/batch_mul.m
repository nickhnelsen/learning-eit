% Batch multiplication of x times y, where:
% x (N_1,N_2) is scalar-valued
% y (K_1,K_2,...,K_J) is scalar-valued with J batch dimensions

function out = batch_mul(x,y)
% out (N_1,N_2,K_1*K_2*...*K_J) is scalar-valued with a single lumped batch dimension
out = squeeze(x.*reshape(y,1,1,[]));


