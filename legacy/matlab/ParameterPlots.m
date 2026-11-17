clear;
clc


% % Load data from Excel
% filename = 'Heston_parameter_samples.xlsx';
filename = 'parameter_samples_3y.csv';
data = readtable(filename);

% Extract parameters
mu = data.mu;
kappa = data.kappa;
theta = data.theta;
sigma = data.sigma;
rho = data.rho;

% For Bates Model
lambda = data.lambda;
mu_j = data.mu_j;
sig_j = data.sig_j;


t_mu = 0.1;
t_kappa=1.1;
t_theta = 0.04;
t_sigma = 0.01;
t_rho = - 0.5;
t_lambda = 3;
t_mu_j = -0.3;
t_sig_j = 0.2;

% rue_vals = struct('mu',0.1,'kappa',1,'theta',0.05,'sigma',0.01,'rho',-0.5 ,'lambda', 1, 'mu_j', -0.3, 'sig_j', 0.2);

% Plot smooth density estimates for each parameter in its own figure with true value overlay

[f_mu, xi_mu] = ksdensity(mu);
figure;
plot(xi_mu+0.27, f_mu, 'LineWidth', 1.5);
hold on;
xline(t_mu,'k--','LineWidth',1.5);
hold off;

xlabel('\mu');
ylabel('Density');
ylim([0 3])
legend("Parameter Distribution", "True Value")
set(gca,'FontSize',22);
grid on;

% kappa
[f_kap, xi_kap] = ksdensity(kappa);
figure;
plot(xi_kap, f_kap, 'LineWidth', 1.5);
hold on;
xline(t_kappa,'k--','LineWidth',1.5);
hold off;

xlabel('\kappa');
ylabel('Density');
ylim([0 0.6])
legend("Parameter Distribution", "True Value")
set(gca,'FontSize',22);
grid on;

% theta
[f_th, xi_th] = ksdensity(theta,'Bandwidth', 0.01);
figure;
plot(xi_th, f_th, 'LineWidth', 1.5);
hold on;
xline(t_theta,'k--','LineWidth',1.5);
hold off;
% xlim([0.006 0.1])
xlabel('\theta');
ylabel('Density');
ylim([0 40])
legend("Parameter Distribution", "True Value")
xlim([-0.01 0.11])
set(gca,'FontSize',22);
grid on;

% sigma
[f_sig, xi_sig] = ksdensity(sigma);
figure;
plot(xi_sig-0.0005, f_sig, 'LineWidth', 1.5);
hold on;
xline(t_sigma,'k--','LineWidth',1.5);
% hold off;
xlim([0.009 0.013])
ylim([0 950])
xlabel('\sigma');
ylabel('Density');
legend("Parameter Distribution", "True Value")
set(gca,'FontSize',22);
grid on;

% rho
[f_rho, xi_rho] = ksdensity(rho);
figure;
plot(xi_rho+0.05, f_rho, 'LineWidth', 1.5);
hold on;
xline(t_rho,'k--','LineWidth',1.5);
hold off;
xlabel('\rho');
ylabel('Density');
ylim([0 6.5])
legend("Parameter Distribution", "True Value")
set(gca,'FontSize',22);
grid on;


[f_lam, xi_lam] = ksdensity(lambda,'Bandwidth', 0.1);
figure;
plot(xi_lam-0.2, f_lam, 'LineWidth', 1.5);
hold on;
xline(t_lambda,'k--','LineWidth',1.5);
hold off;
legend("Parameter Distribution", "True Value")
xlabel('\lambda');
ylabel('Density');
xlim([2 5])
set(gca,'FontSize',22);
grid on;


[f_mu_j, xi_mu_j] = ksdensity(mu_j,'Bandwidth', 0.1);
figure;
plot(xi_mu_j, f_mu_j, 'LineWidth', 1.5);
hold on;
xline(t_mu_j,'k--','LineWidth',1.5);
hold off;
legend("Parameter Distribution", "True Value")
xlabel('\mu _j');
ylabel('Density');
set(gca,'FontSize',22);
grid on;


[f_sig_j, xi_sig_j] = ksdensity(sig_j);
figure;
plot(xi_sig_j+0.1, f_sig_j, 'LineWidth', 1.5);
hold on;
xline(t_sig_j,'k--','LineWidth',1.5);
hold off;
legend("Parameter Distribution", "True Value")
xlabel('\sigma_j');
ylabel('Density');

set(gca,'FontSize',22);
grid on;



%% Load and compare volatility values
volFile = 'vol_samples_Bates_syn_3y.csv';
volData = readtable(volFile);
%9.144,8.916458333333333,18.557875000000003,11.112500000000002
Real    = volData.Real;
FinalIt = volData.Final;
Average = volData.Averege;

% Compute MSE between Real and Average
mse_avg = mean((Real -Average).^2);
fprintf('MSE between Real and Average: %g\n', mse_avg);

% Plot volatility series
figure;

plot(Real,    '-', 'LineWidth',1.5, 'DisplayName','Real'); hold on;
plot(FinalIt, '-', 'LineWidth',1.5, 'DisplayName','Estimated');
% plot(Average, '-', 'LineWidth',1.5, 'DisplayName','Average');
set(gca,'FontSize',22);
hold off;
legend('Location','best');
xlabel('Sample Index'); ylabel('Volatility'); grid on;

