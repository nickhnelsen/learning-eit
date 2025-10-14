% Smooth cutoff function
% Reference:  P. D. Miller, Applied Asymptotic Analysis, Graduate Studies in Math, 75 (2006), AMS.

function y = cutoff(x, Valp, Rm, Rp, scale)
y = 0.5*(1+Valp) + 0.5*(1-Valp).*tanh((1./(x - Rm) + 1./(x - Rp))/scale);
y(x<=Rm)=1;
x=(x>=Rp);
y(x)=Valp(x);














