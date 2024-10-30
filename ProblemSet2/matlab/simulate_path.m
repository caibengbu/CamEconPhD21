function [a_path_sim, z_path_sim] = simulate_path(joint_transition, N, T, params)
    rng(0); % Set the seed for reproducibility
    % simulate_path - Simulate the path of the economy
    %
    % Inputs:
    %   joint_transition - Joint transition matrix
    %   initial_state    - Initial state of the economy
    %   T                - Number of periods to simulate
    %
    % Outputs:
    %   a_path_sim       - Simulated path of assets
    %   z_path_sim       - Simulated path of productivity
    %   joint_path_sim   - Simulated path of joint distribution
    % Initialize paths
    a_path_sim_ind = zeros(N, T, "int32");
    z_path_sim_ind = zeros(N, T, "int32");
    a_grid = get_a_grid(params);
    z_grid = exp(tauchen(params));

    % Set initial state
    joint_stationary_distribution = stationary_distribution(joint_transition);
    current_state = randsample(1:size(joint_transition, 1), N, true, joint_stationary_distribution);

    for t = 1:T
        % Record current state
        [a_ind, z_ind] = ind2sub([params.Na, params.Nz], current_state);
        a_path_sim_ind(:,t) = a_ind;
        z_path_sim_ind(:,t) = z_ind;

        new_state = zeros(1, N, "int32");
        % Draw next state
        for i = 1:N
            new_state(i) = randsample(1:size(joint_transition, 1), 1, true, joint_transition(current_state(i), :));
        end
        current_state = new_state;
        fprintf("Simulating period %d\n", t);
    end

    a_path_sim = a_grid(a_path_sim_ind);
    z_path_sim = z_grid(z_path_sim_ind);
end