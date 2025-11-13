% -----------------------------------------------
% MATLAB script: load S&P 500 returns & plot
% -----------------------------------------------
clear; clc; close all;

% Read log returns
T = readtable('sp500_returns.csv');
returns = T.LogReturn;
dates = datetime(T.Date);

% --- Histogram vs Normal Distribution
figure;
histogram(returns, 100, 'Normalization', 'pdf', 'DisplayName','Empirical');
hold on;
xvals = linspace(min(returns), max(returns), 1000);
mu = mean(returns); sigma = std(returns);
plot(xvals, normpdf(xvals, mu, sigma), 'r-', 'LineWidth', 2, 'DisplayName','Normal Approximation');
xlabel('Return'); ylabel('Probability Density'); legend; grid on;
set(gca,'FontSize',16,'LineWidth',1.2);
% --- Rolling Volatility (21-day window)
window = 7;
rollVol = movstd(returns, window) * sqrt(252);  % Annualized
figure;
plot(dates(window:end), rollVol(window:end), 'b');
xlabel('Date'); ylabel('Volatility'); grid on;
set(gca,'FontSize',14,'LineWidth',1.2);