clear; clc; close all;

% Parameters
T = 1;              % Time horizon (e.g. 1 year)
N = 500;            % Number of time steps
dt = T/N;           % Time increment
t = linspace(0, T, N+1);

M = 30;             % Number of sample paths
mu = 0.1;           % Drift (for GBM)
sigma = 0.2;        % Volatility
S0 = 100;           % Initial asset price (for GBM)

rng(232); % For reproducibility

% Preallocate
BM = zeros(M, N+1);        % Brownian motion paths
GBM = zeros(M, N+1);       % Geometric Brownian motion paths

for i = 1:M
    dW = sqrt(dt) * randn(1, N);         % Brownian increments
    W = [0, cumsum(dW)];                 % Brownian path
    BM(i, :) = W;
    GBM(i, :) = S0 * exp((mu - 0.5*sigma^2)*t + sigma*W); % GBM path
end

% Plot Brownian Motion paths
figure('Name','Brownian Motion','NumberTitle','off');
plot(t, BM', 'LineWidth', 1.2);
xlabel('Time'); ylabel('B(t)');
set(gca,'FontSize',16,'LineWidth',1.2);
grid on;

% Plot Geometric Brownian Motion paths
figure('Name','Geometric Brownian Motion','NumberTitle','off');
plot(t, GBM', 'LineWidth', 1.2);
xlabel('Time'); ylabel('S(t)');
set(gca,'FontSize',16,'LineWidth',1.2);
grid on;
