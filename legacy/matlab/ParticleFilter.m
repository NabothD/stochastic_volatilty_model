%% MATLAB: Kalman Filter vs SIR Particle Filter for Volatility Estimation
clear all;
clc


% Set seed for reproducibility
rng(1034);

% Simulation parameters
T = 200;         % Number of time steps
dt = 1/252;      % Daily observations (typical trading year)

% True volatility model parameters
kappa = 1;       % Mean reversion speed
theta = 0.04;    % Long-run variance
sigma_v = 0.2;   % Volatility of volatility

% Pre-allocate
v_true = zeros(1,T);
S = zeros(1,T);

% Initial values
v_true(1) = theta;
S(1) = 100;

% Generate synthetic volatility and price
for t = 2:T
    v_true(t) = v_true(t-1) + kappa*(theta - v_true(t-1))*dt + sigma_v*sqrt(v_true(t-1)*dt)*randn;
    S(t) = S(t-1)*exp(-0.5*v_true(t-1)*dt + sqrt(v_true(t-1)*dt)*randn);
end

%% Kalman Filter Setup (linear approximation)
Q = sigma_v^2*dt; % Process variance
R = var(diff(log(S))); % Measurement variance (observed)

% Initialize Kalman filter
v_kalman = zeros(1,T);
P = 0.01;
v_kalman(1) = theta;

for t = 2:T
    % Prediction
    v_pred = v_kalman(t-1) + kappa*(theta - v_kalman(t-1))*dt;
    P_pred = P + Q;
    
    % Update
    K = P_pred / (P_pred + R);
    obs_vol = (log(S(t)/S(t-1)))^2/dt; % Squared returns as proxy
    v_kalman(t) = v_pred + K*(obs_vol - v_pred);
    P = (1-K)*P_pred;
end

%% SIR Particle Filter Setup
N_particles = 10000;
v_particles = theta * ones(N_particles,1);
weights = ones(N_particles,1)/N_particles;

v_SIR = zeros(1,T);
v_SIR(1) = theta;

for t = 2:T
    % Propagate particles
    v_particles = v_particles + kappa*(theta - v_particles)*dt + sigma_v*sqrt(max(v_particles,0)*dt).*randn(N_particles,1);
    
    % Measurement likelihood
    obs_vol = (log(S(t)/S(t-1)))^2/dt;
    likelihoods = exp(-0.5*((obs_vol - v_particles).^2)/(R)) + eps;
    weights = likelihoods .* weights;
    weights = weights / sum(weights);

    % Resample
    indices = randsample(N_particles,N_particles,true,weights);
    v_particles = v_particles(indices);
    weights = ones(N_particles,1)/N_particles;

    % Estimate
    v_SIR(t) = mean(v_particles);
end

%% Plotting results
figure;
plot(1:T, sqrt(v_true)*100,'k','LineWidth',2); hold on;
plot(1:T, sqrt(v_kalman)*100,'b--','LineWidth',1.5);
plot(1:T, sqrt(v_SIR)*100,'r-.','LineWidth',1.5);
legend('True Volatility','Kalman Filter Estimation','SIR Particle Filter Estimation');
xlabel('Time Steps'); ylabel('Volatility (%)');
title('Kalman Filter vs SIR Particle Filter Volatility Estimation');
grid on;
