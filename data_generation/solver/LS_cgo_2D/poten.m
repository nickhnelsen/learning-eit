% Evaluates the Schrodinger potential corresponding to conductivity
% sigma.m. Uses finite differences to implement the Laplace operator.
%
% Arguments:
% z 		evaluation points, complex.
%
% Calls to:
% sigma.m

% function out = poten(z)
% sg0 = sqrt(sigma(z));
% dx = z(2,2)-z(1,1);
% dy = abs(imag(dx));
% dx = abs(real(dx));
% out = 4*del2(sg0,dx,dy)./sg0;

function out = poten(z)

h   = 1e-6; % 1e-6 causes artifacts, 1e-2 better for random sigma
z0  = z;
z1  = z+h;
z2  = z+1i*h;
z3  = z-h;
z4  = z-1i*h;
sg0 = sqrt(sigma(z0));
sg1 = sqrt(sigma(z1));
sg2 = sqrt(sigma(z2));
sg3 = sqrt(sigma(z3));
sg4 = sqrt(sigma(z4));

out = 1/(h^2)*(sg1 + sg2 + sg3 + sg4 - 4*sg0)./sg0;


