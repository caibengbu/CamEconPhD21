function [r_tmp, a_stationary_distribution, z_stationary_distribution, joint_transition, v_res, policy_res] = naive_solver(params)
    % Set default tolerance
    tol = 1e-6;

    % Unpack parameters
    A = params.A;
    alpha = params.alpha;
    delta = params.delta;
    Na = params.Na;
    Nz = params.Nz;

    r_tmp = 0.0319; % initial guess

    converged = false;
    maxiter = 1000;
    i = 0;
    stepsize = 0.1;

    a_stationary_distribution = zeros(Na, 1);
    z_stationary_distribution = zeros(1, Nz);
    v_res = zeros(Na, Nz);

    [lgz_grid, P] = tauchen(params);
    z_grid = exp(lgz_grid);
    a_grid = get_a_grid(params);

    % guess a "K" instead of r. because K is predetermined of the past period
    % as an information set for the consumer
    % especially in RBC model

    while i < maxiter && ~converged
        w_tmp = A * (1 - alpha) * (A * alpha / (r_tmp + delta))^(alpha / (1 - alpha));
        [v_res, policy_res] = vfi(w_tmp, r_tmp, params, v_res);

        joint_transition = cell(Nz, Nz);
        for z_now_ind = 1:Nz
            for z_next_ind = 1:Nz
                z_transition = P(z_now_ind, z_next_ind);
                a_transition_mat_given_z = zeros(Na, Na);
                for j = 1:Na
                    policy = policy_res{j, z_now_ind};
                    a_next_min_ind = policy(1);
                    a_next_min_ind_weight = policy(2);
                    a_next_max_ind = policy(3);
                    a_next_max_ind_weight = policy(4);
                    a_transition_mat_given_z(j, a_next_min_ind) = a_next_min_ind_weight;
                    if a_next_max_ind ~= a_next_min_ind
                        a_transition_mat_given_z(j, a_next_max_ind) = a_next_max_ind_weight;
                    end
                end
                % need to check the joint transition adds up to one(Na*Nz, 1)
                joint_transition{z_now_ind, z_next_ind} = z_transition * a_transition_mat_given_z;
            end
        end

        joint_transition = cell2mat(joint_transition);
        % or I can just do iterative method to find fixed point for a'P = a
        % directly compute the (a,z) distribution at t+1, instead of constructing this joint transition matrix
        az_stationary_distribution = stationary_distribution(joint_transition);
        az_stationary_distribution = reshape(az_stationary_distribution, Na, Nz);
        a_stationary_distribution = sum(az_stationary_distribution, 2);
        z_stationary_distribution = sum(az_stationary_distribution, 1);
        K = a_stationary_distribution' * a_grid';
        L = z_stationary_distribution * z_grid';

        r_new = alpha * A * (K / L)^(alpha - 1) - delta;
        err = abs(r_new - r_tmp);
        if err < tol
            converged = true;
        else
            i = i + 1;
            fprintf('Iteration %d, error: %f, r: %f\n', i, err, r_tmp);
            r_tmp = stepsize * r_new + (1 - stepsize) * r_tmp;
        end
    end

    if ~converged
        error('Did not converge');
    end

    policy_res = cellfun(@(policy) ...
        a_grid(policy(1)) * policy(2) + a_grid(policy(3)) * policy(4), ...
        policy_res);
end