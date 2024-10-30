function [z, P] = tauchen(params)
    % tauchen - Approximates an AR(1) process using Tauchen's method
    %
    % Inputs:
    %   N    - Number of grid points
    %   mu   - Mean of the AR(1) process
    %   rho  - AR(1) coefficient
    %   sigma- Standard deviation of the error term
    %   m    - Scaling parameter for the grid range
    %
    % Outputs:
    %   z    - Grid points
    %   P    - Transition probability matrix

    % Step 1: Discretize the state space

    % Unpack parameters
    N = params.Nz;
    mu = 0;
    rho = params.rho;
    sigma = params.sigma;
    m = params.m;

    z = linspace(mu - m * sqrt(sigma^2 / (1 - rho^2)), ...
                 mu + m * sqrt(sigma^2 / (1 - rho^2)), N);

    % Step 2: Calculate the width of each grid interval
    z_step = z(2) - z(1);

    % Step 3: Initialize the transition probability matrix
    P = zeros(N, N);

    % Step 4: Fill the transition probability matrix
    for j = 1:N
        for k = 1:N
            if k == 1
                P(j, k) = normcdf((z(1) - mu - rho * z(j) + z_step / 2) / sigma);
            elseif k == N
                P(j, k) = 1 - normcdf((z(N) - mu - rho * z(j) - z_step / 2) / sigma);
            else
                P(j, k) = normcdf((z(k) - mu - rho * z(j) + z_step / 2) / sigma) - ...
                          normcdf((z(k) - mu - rho * z(j) - z_step / 2) / sigma);
            end
        end
    end
end