clear variables;

N=1024;
tau=20;
alpha=4.5;
rho=0.75;
contrast_ratio=30; % 30
sigma=25;

tau_lnt=8;
alpha_lnt=3.6;
rhop_lnt=0.95;
rhom_lnt=0.5;
scale=8;

addpath ..;
% [~, sigma_interp] = get_cond_data_smooth(N,tau,alpha,rho,contrast_ratio,sigma);
[~, sigma_interp] = get_cond_data_lognormaltrunc(N,tau_lnt,alpha_lnt,rhop_lnt,rhom_lnt,scale);
rmpath ..;

save data/single_sigma sigma_interp
