%% Simulating Heston Model with and without Merton Jumps
clear; clc; 

% Parameters for Heston Model
S0 = 100;         % Initial asset price
r = 0.1;         % Risk-free rate
T = 3;            % Time in years
N = 252*3;          % Number of time steps (days)

% Heston Parameters
kappa = 1.1;        % Rate of mean reversion
theta = 0.04;     % Long-term variance
sigma = 0.01;     % Volatility of variance (vol of vol)
rho = -0.5;       % Correlation between asset and variance
v0 = 0.04;        % Initial variance

% Jump parameters (Merton)
lambda = 3;       % Jump intensity (jumps per year)
muJ = -0.3;       % Mean of jump size
sigmaJ = 0.2;     % Standard deviation of jump size

% Simulation time grid
dt = T/N;
time = linspace(0, T, N+1);

%% Preallocate arrays
S_heston = zeros(1, N+1);
S_jump   = zeros(1, N+1);
v        = zeros(1, N+1);

S_heston(1) = S0;
S_jump(1)   = S0;
v(1)        = v0;

%% Generate correlated Brownian motions
Z1 = randn(1, N);
Z2 = rho * Z1 + sqrt(1 - rho^2) * randn(1, N);

%% Simulate price paths
for t = 1:N
    % Heston variance update (ensuring non-negativity)
    v(t+1) = abs( v(t) + kappa*(theta - v(t))*dt + sigma*sqrt(v(t)*dt)*Z2(t) );
    
    % Pure Heston (no jumps) for reference
    S_heston(t+1) = S_heston(t) * exp( (r - 0.5*v(t))*dt + sqrt(v(t)*dt)*Z1(t) );
    
    % Heston with jumps (Merton jumps)
    Njump = poissrnd(lambda * dt);  % Number of jumps in this time step
    if Njump > 0
        JumpSize = sum( muJ + sigmaJ * randn(1, Njump) );
    else
        JumpSize = 0;
    end
    
    S_jump(t+1) = S_jump(t) * exp( (r - 0.5*v(t))*dt + sqrt(v(t)*dt)*Z1(t) + JumpSize );
end

%% Save results to CSV file
% Create a table with columns: Time, Price (with jumps), Variance
T_data = table(time', S_jump', v', 'VariableNames', {'Time', 'Price', 'Variance'});
writetable(T_data, 'prices_with_jumps.csv');

%% Plot results
figure;
plot(time, S_heston, 'LineWidth', 1.5, 'Color', 'k'); hold on;
plot(time, S_jump, '-.', 'LineWidth', 1.5, 'Color', 'r');
grid on;
xlabel('Time (Years)');
ylabel('Asset Price');
legend('Heston','Bates','Location','best');
set(gca,'FontSize',14,'LineWidth',1.2);
