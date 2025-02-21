clear all; close all; clc;

%% Parameters
q = 0.9; % Fractional order (0 < q < 1)
T = 5; % Total time
dt = 0.5; % Time step
t = 0:dt:T; % Time grid
N = length(t); % Number of time steps

% System parameters (Lotka-Volterra)
a = 0.8; b = 0.5; c = 0.3; d = 0.2;
f = @(x) [x(1)*(a - b*x(2)); x(2)*(c*x(1) - d)]; % Nonlinear dynamics

% Kernel (Gaussian)
sigma = 0.3; % Bandwidth
K = @(x,y) exp(-norm(x - y)^2 / (2*sigma^2));

%% Step 1: Simulate Original System
x0 = [1.5; 0.5]; % Initial condition
X = zeros(2, N); % True states
X(:,1) = x0;

% ABM Predictor-Corrector for Caputo fractional system
for n = 1:N-1
    % Predictor
    x_pred = X(:,n) + (dt^q/gamma(q+1)) * f(X(:,n));
    
    % Corrector
    sum_hist = 0;
    for j = 1:n
        weight = ((n-j+1)^q - (n-j)^q);
        sum_hist = sum_hist + weight * f(X(:,j));
    end
    X(:,n+1) = X(:,1) + (dt^q/gamma(q+2)) * (f(x_pred) + sum_hist);
end

%% Step 2: DMD Setup (Simplified Coefficients)
% Occupation kernels for trajectories
Gamma = @(tau, x) (T - tau).^(q-1) .* K(x, X(:,round(tau/dt)+1));

%% Step 2: DMD Setup (Updated Coefficients)
alpha = q;  % Set alpha equal to q

% Calculate a_tilde coefficients according to the formula
a_tilde = zeros(N, N);  % Matrix where each column k+1 contains coefficients for time t_{k+1}

for k = 0:N-1  % For each t_{k+1}
    t_k1 = t(k+1);
    
    % i = 0 case
    a_tilde(1, k+1) = ((t_k1 - t(1))^alpha)/alpha;
    
    % 1 ≤ i ≤ k cases
    for i = 1:k
        a_tilde(i+1, k+1) = ((t_k1 - t(i))^alpha)/alpha - ((t_k1 - t(i+1))^alpha)/alpha;
    end
    
    % i = k+1 case
    if k < N-1
        a_tilde(k+2, k+1) = ((t_k1 - t(k+1))^alpha)/alpha;
    end
end
%% Step 3: Build Gram Matrix (G) and Operator Matrix (A)
G = zeros(N, N);
A = zeros(N, N);

for i = 1:N
    for j = 1:N
        % Gram matrix entry using updated a_tilde
        G(i,j) = 0;
        for k = 1:N
            G(i,j) = G(i,j) + a_tilde(i,k) * a_tilde(j,k) * K(X(:,i), X(:,j));
        end
        
        % Operator matrix entry
        term = K(X(:,i), X(:,end)) - K(X(:,i), X(:,1));
        A(i,j) = sum(a_tilde(i,:)) * term;
    end
end


%% Step 4: DMD Eigenvalue Problem
[V, D] = eig(A, G);     % Generalized eigenvalues
lambda = diag(D);      % DMD eigenvalues
Phi = V;                % DMD modes

%% step 5: Reconstruction approach

% Project initial condition onto DMD modes
b = zeros(N,1);
for i = 1:N
    b(i) = a_tilde(1,i) * K(X(:,1), X(:,i));
end

% Reconstruction using Mittag-Leffler (approximated)
X_rec = zeros(2, N);
for k = 1:N
    temp = zeros(2,1);
    for i = 1:N
        for j = 1:N
            temp = temp + Phi(i) * V(j) * b(j) * K(X(:,i), X(:,1)) * mlf(q, 1, lambda(j) * t(k)^q, 7);
        end
    end
    X_rec(:,k) = temp;
end

%% Step 6: Plot Results
figure;
subplot(2,1,1);

%plot(t, X_rec(1,:), 'b', 'LineWidth', 2);
plot(t, X(1,:), 'b', t, X_rec(1,:), 'r--', 'LineWidth', 2);
legend('True x_1', 'Reconstructed x_1');
title('State 1 Comparison');

subplot(2,1,2);
%plot(t, X(2,:), 'b', 'LineWidth', 2);
plot(t, X(2,:), 'b', t, X_rec(2,:), 'r--', 'LineWidth', 2);
legend('True x_2', 'Reconstructed x_2');
title('State 2 Comparison');

% Error plot
figure;
error = vecnorm(X - X_rec);
plot(t, error, 'k', 'LineWidth', 2);
title('Reconstruction Error');
xlabel('Time'); ylabel('||X - X_{rec}||');


%% Mittag-Leffler Function (Approximation)
function E = mlf(alpha, beta, z, kmax)
% Approximate Mittag-Leffler function using truncated series
E = zeros(size(z));
for k = 0:kmax
    E = E + z.^k ./ gamma(alpha*k + beta);
end
end
