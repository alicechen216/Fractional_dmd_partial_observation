clear; clc; close all; 


%% System Parameters
alpha = 0.5;        % Fractional order
a = 2;              % Coefficient in the equation
y0 = 2;             % Initial condition
obs_ratio = 0.2;    % 20% spatial observation
rng(42);            % Reproducible random sampling

%% Space-Time Parameters
tf = 10; dt = 0.001;
t = 0:dt:tf;
xi = linspace(-10,10,100);
[Xgrid,T] = meshgrid(xi,t);

%% Analytical Solution
lambda = -1/(2);    % From 2D^{0.5}y + y = 0
y_analytical = y0 * mlf(alpha, 1, lambda*t.^alpha, 6);
y = cos(Xgrid') .* y_analytical;  % f(x,t) = cos(x)y(t)
%y=y_analytical;

%% Partial Observation Setup
n_obs = floor(length(xi)*obs_ratio);
obs_indices = sort(randperm(length(xi),n_obs));
y_obs = y(obs_indices,:);      % Observed data

%% DMD Parameters
lod = 0.2;          % Use first 20% temporal data
y1_obs = y_obs(:,1:floor(lod*length(t)));
r = 1;              % Rank truncation

%% =================================================================
%% F-DMD with Partial Observations
%% =================================================================
% Feature matrix construction
memory = floor(1*length(y1_obs));  % Full memory
w_obs = foweight(alpha-1, memory, n_obs);
dy1_obs = creatfeature(y1_obs(:,2:end)-y1_obs(:,1:end-1), memory, 1);
nablaY_obs = w_obs * dy1_obs;

% Dimension reduction
X_obs = y1_obs(:,1:end-1);
[U_obs,S_obs,V_obs] = svd(X_obs,'econ');
U_r_obs = U_obs(:,1:r); S_r_obs = S_obs(1:r,1:r); V_r_obs = V_obs(:,1:r);

% F-DMD operator
Atilde_obs = U_r_obs' * nablaY_obs * V_r_obs / S_r_obs;
[W_obs,D_obs] = eig(Atilde_obs);
Phi_obs = U_r_obs * W_obs;
b_obs = pinv(Phi_obs) * X_obs(:,1);

% Spatial interpolation
Phi_full = zeros(length(xi),r);
for k = 1:r
    Phi_full(:,k) = interp1(xi(obs_indices), Phi_obs(:,k), xi, 'spline');
end

% Reconstruction
y2_full = zeros(length(xi),length(t));
for i = 1:length(t)
    y2_full(:,i) = Phi_full * ml_matrix(D_obs*t(i)^alpha/dt^alpha, alpha,1) * b_obs;
end

%% =================================================================
%% Corrected Exact DMD Section
%% =================================================================
% Exact DMD with Partial Observations (Fixed Dimensions)
X_exact_obs = y1_obs(:,1:end-1);
Y_exact_obs = y1_obs(:,2:end);

% Use pseudoinverse to handle rank deficiency
[U_exact,S_exact,V_exact] = svd(X_exact_obs,'econ');
r_exact = rank(S_exact); % Dynamic rank selection
if r_exact < r
    warning('Reducing rank from %d to %d due to numerical rank', r, r_exact);
    r = r_exact;
end

Atilde_exact_obs = U_exact(:,1:r)' * Y_exact_obs * V_exact(:,1:r) / S_exact(1:r,1:r);
[W_exact_obs,D_exact_obs] = eig(Atilde_exact_obs);
omega_exact_obs = log(diag(D_exact_obs))/dt;

% Mode interpolation with dimension check
Phi_exact_full = zeros(length(xi),r);
for k = 1:r
    Phi_exact_full(:,k) = interp1(xi(obs_indices), U_exact(:,k), xi, 'spline');
end

% Correct amplitude calculation
b_exact_obs = pinv(Phi_exact_full(obs_indices,:)) * X_exact_obs(:,1);

%% =================================================================
%% Fixed Reconstruction (Both Methods)
%% =================================================================
% F-DMD Reconstruction (unchanged)
y2_full = zeros(length(xi),length(t));
for i = 1:length(t)
    y2_full(:,i) = Phi_full * ml_matrix(D_obs*t(i)^alpha/dt^alpha, alpha,1) * b_obs;
end

% Corrected Exact DMD Reconstruction
 y_exact_full = Phi_exact_full * (exp(omega_exact_obs*t) .* b_exact_obs);

figure(2)
plot(t, real(y(1,:)), 'LineWidth', 2)
hold on
plot(t, real(y2_full(1,:)), '--', 'LineWidth', 1.5)
plot(t, real(y_exact_full(1,:)), 'b:', 'LineWidth', 1.5)
xline(lod*tf, 'k--', 'Training End')
legend('Analytical', 'F-DMD', 'Exact DMD')
title('Time Evolution'), xlabel('Time'), ylabel('f(x=0,t)')
%% =================================================================
%% Visualization (Consistent Style)
%% =================================================================
figure(3)
%set(gcf,'Position',[100 100 1200 400])

% Analytical Solution
subplot(1,3,1)
surf(Xgrid, T, real(y'), 'EdgeColor','none')
shading interp
title('Analytical Solution')
xlabel('x'); ylabel('t'); zlabel('f(x,t)')
colormap(parula)
view(-40,30)
axis tight

% F-DMD Reconstruction
subplot(1,3,2)
surf(Xgrid, T, real(y2_full'), 'EdgeColor','none')
shading interp
title('FDMD Reconstruction (20% Obs)')
xlabel('x'); ylabel('t'); zlabel('f(x,t)')
colormap(hot)
view(-40,30)
axis tight

% Exact DMD Reconstruction
subplot(1,3,3)
surf(Xgrid, T, real(y_exact_full'), 'EdgeColor','none')
shading interp
title('DMD Reconstruction (20% Obs)')
xlabel('x'); ylabel('t'); zlabel('f(x,t)')
colormap(cool)
view(-40,30)
axis tight
exportgraphics(gcf, 'figure3.png', 'Resolution', 300); % 300 DPI
%% Time Slice Comparison
figure(4)
set(gcf,'Position',[100 100 800 600])

% Spatial profile at t=5
t_slice = 5;
[~,t_idx] = min(abs(t - t_slice));

subplot(2,1,1)
plot(xi, real(y(:,t_idx)), 'LineWidth',3)
hold on
plot(xi, real(y2_full(:,t_idx)), '--','LineWidth',2.5)
plot(xi, real(y_exact_full(:,t_idx)), ':','LineWidth',2.5)
title(['Spatial Profile at t = ' num2str(t_slice)])
xlabel('x'); ylabel('f(x,t)')
legend('Analytical','FDMD','DMD','Location','best')
grid on

% Temporal evolution at x=0
x_slice = 0;
[~,x_idx] = min(abs(xi - x_slice));

subplot(2,1,2)
plot(t, real(y(x_idx,:)), 'LineWidth',3)
hold on
plot(t, real(y2_full(x_idx,:)), '--','LineWidth',2.5)
plot(t, real(y_exact_full(x_idx,:)), ':','LineWidth',2.5)
title(['Temporal Evolution at x = ' num2str(x_slice)])
xlabel('t'); ylabel('f(x,t)')
legend('Analytical','FDMD','DMD','Location','best')
grid on

%% Error Metrics
fprintf('F-DMD Relative Error: %.4f\n', norm(y2_full-y)/norm(y));
fprintf('Exact DMD Relative Error: %.4f\n', norm(y_exact_full-y)/norm(y));
%% Corrected creatfeature function
function xnew = creatfeature(x, L, numb)
    [m, n] = size(x);
    xnew = zeros(L * m, n);  % Correct initialization
    for i = 1:m
        r = zeros(1, L);
        r(1:numb) = x(i, 1:numb);
        current_block = toeplitz(r, x(i, :));
        xnew(L*(i-1)+1:L*i, :) = current_block;
    end
end

function w=foweight(alpha,L,r)
w1=1;
w=zeros(r,L);
for i=2:L
w1(i)=w1(i-1)*(1-(alpha+1)/(i-1));
end
cells = repmat({w1}, 1, r);
w=blkdiag(cells{:});
end