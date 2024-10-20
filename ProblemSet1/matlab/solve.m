% Exogeneous variables
a = 8.0;
alpha = 0.3;
tau = 0.15;
A = 1.0;
r = 0.04;
beta = 0.96;
chi = 1.0;

target = [0.33, 0.06, 0.25, 0.70];
weights = eye(4);

x0 = [-1.0, -1.0, -0.5];
lb = [0.0, 0.0, 0.0];
ub = [Inf, Inf, Inf];
f = @(x) loss_function(a,exp(x(1)),r,tau,exp(x(2)),chi,beta,exp(x(3)), alpha,A,target,weights);
options = optimset('Display','iter');
[xmin, fval] = fminsearch(f, x0, options);

b = exp(xmin(1)); % 0.1230
eta = exp(xmin(2)); % 1.1507
sigma_z = exp(xmin(3)); % 0.2996
