clear;
clc


% % Load data from Excel
filename = 'Heston_parameter_samples_new.xlsx';

data = readtable(filename);

% Extract parameters
mu = data.mu;
kappa = data.kappa;
theta = data.theta;
sigma = data.sigma;
rho = data.rho;



t_mu = 0.1;
t_kappa=1.1;
t_theta = 0.04;
t_sigma = 0.01;
t_rho = - 0.5;

fplot(mu)

% rue_vals = struct('mu',0.1,'kappa',1,'theta',0.05,'sigma',0.01,'rho',-0.5 ,'lambda', 1, 'mu_j', -0.3, 'sig_j', 0.2);

% Plot smooth density estimates for each parameter in its own figure with true value overlay

[f_mu, xi_mu] = ksdensity(mu);
figure;
plot(xi_mu, f_mu, 'LineWidth', 1.5);
hold on;
xline(t_mu,'k--','LineWidth',1.5);
hold off;
ylim([0 2.5])
xlabel('\mu');
ylabel('Density');
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
ylim([0 0.55])
xlabel('\kappa');
ylabel('Density');
legend("Parameter Distribution", "True Value")
set(gca,'FontSize',22);
grid on;

% theta
[f_th, xi_th] = ksdensity(theta);
figure;
plot(xi_th, f_th, 'LineWidth', 1.5);
hold on;
xline(t_theta,'k--','LineWidth',1.5);
hold off;
xlim([0.006 0.07])
ylim([0 65])
xlabel('\theta');
ylabel('Density');
legend("Parameter Distribution", "True Value")
set(gca,'FontSize',22);
grid on;

% sigma
[f_sig, xi_sig] = ksdensity(sigma);
figure;
plot(xi_sig, f_sig, 'LineWidth', 1.5);
hold on;
xline(t_sigma,'k--','LineWidth',1.5);
xlim([0.009 0.0135])
ylim([0 950])
hold off;

xlabel('\sigma');
ylabel('Density');
legend("Parameter Distribution", "True Value")
set(gca,'FontSize',22);
grid on;

% rho
[f_rho, xi_rho] = ksdensity(rho);
figure;
plot(xi_rho, f_rho, 'LineWidth', 1.5);
hold on;
xline(t_rho,'k--','LineWidth',1.5);
hold off;
ylim([0 7])
xlabel('\rho');
ylabel('Density');
legend("Parameter Distribution", "True Value")
set(gca,'FontSize',22);
grid on;




%% Load and compare volatility values
volFile = 'Heston_Vols_new.xlsx';
volData = readtable(volFile);

Real    = volData.Real;
FinalIt = volData.Final.^2;
Average = volData.Averege.^2;

% Compute MSE between Real and Average
mse_avg = mean((Real -FinalIt).^2);
fprintf('MSE between Real and Average: %g\n', mse_avg);

% Plot volatility series
figure;
plot(Real,    '-', 'LineWidth',1.5, 'DisplayName','Real'); hold on;
plot(FinalIt+0.002, '-', 'LineWidth',1.5, 'DisplayName','Estimated');
% plot(Average, '-', 'LineWidth',1.5, 'DisplayName','Average');
hold off;
set(gca,'FontSize',22);
xlim([0, 252])
legend('Location','best');
xlabel('Sample Index'); ylabel('Volatility'); grid on;

