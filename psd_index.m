clear; close all;
r = [2.85,3.27,5.3,5.7,8.7,9.2,10.4];
alpha_lb = [1.11,1.11,1.13,1.47,1.41,1.43,1.09];
alpha_ub = [1.4,1.26,1.5,1.57,1.63,1.58,1.5];

scatter(r, alpha_lb,'filled','r');
hold on
scatter(r, alpha_ub,'filled','b')
grid on
xlabel('r [Rs]')
ylabel('PSD index')