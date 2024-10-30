function [v_update, policy] = vfi(w, r, params, v_init)

    % Set the tolerance level
    tol = 1e-6;

    % Unpack parameters
    beta = params.beta;
    Nz = params.Nz;
    Na = params.Na;
    a_lower = params.a_lower;
    a_upper = params.a_upper;

    % Get grids and transition matrix
    [lgz_grid, P] = tauchen(params);
    z_grid = exp(lgz_grid);
    a_grid = get_a_grid(params);
    v_array = v_init;

    maxiter = 1000;
    converged = false;
    i = 0;

    v_update = zeros(size(v_array));
    a_next_array = zeros(Na, Nz);

    while i < maxiter && ~converged
        v_array_expectation = v_array * P'; % E_z_next [V[a_next, z_next] | z] = Σ_z V[a_next, z] * P[z_next | z]
        % a function of a_next and z_now
        err = 0.0;
        V_expectation_itp = griddedInterpolant({a_grid, z_grid}, v_array_expectation, 'linear');

        for a_now_ind = 1:Na
            for z_now_ind = 1:Nz
                z_now = z_grid(z_now_ind);
                a_now = a_grid(a_now_ind);
                cash_in_hand = w * z_now + (1 + r) * a_now;
                f = @(a_next_tmp) softlog(cash_in_hand - a_next_tmp) + beta * V_expectation_itp(a_next_tmp, z_now);
                [a_next, v] = fminbnd(@(x) -f(x), a_lower, min(cash_in_hand, a_upper)); % can't save more than cash_in_hand
                v = -v;
                if f(a_lower) > f(a_next)
                    a_next = a_lower;
                    v = f(a_lower);
                end
                a_next_array(a_now_ind, z_now_ind) = a_next;
                % a_lower can be update each iteration. Policy function is increasinsg in a_now. (try to think about it)
                v_update(a_now_ind, z_now_ind) = v;
                v_old = v_array(a_now_ind, z_now_ind);
                err = err + abs(v - v_old);
            end
        end

        if err < tol
            converged = true;
        else
            i = i + 1;
            v_array = v_update;
            fprintf('Iteration %d, error: %f\n', i, err);
        end
    end

    policy = cell(Na, Nz);
    for a_now_ind = 1:Na
        for z_now_ind = 1:Nz
            a_next = a_next_array(a_now_ind, z_now_ind);
            a_next_max_ind = find(a_grid >= a_next, 1); % find the first index that is greater or equal than a_next
            a_next_max = a_grid(a_next_max_ind);
            if a_next_max_ind == 1 % if a_next is exactly zero
                a_next_min_ind = 1; 
                a_next_min_ind_weight = 1.0; % assign full weight to the first element
                a_next_max_ind_weight = 0.0;
            else
                a_next_min_ind = a_next_max_ind - 1;
                a_next_min = a_grid(a_next_min_ind);
                a_next_min_ind_weight = (a_next_max - a_next) / (a_next_max - a_next_min);
                a_next_max_ind_weight = 1 - a_next_min_ind_weight;
            end
            % don't need to store the weight for a_next_max_ind, because it is 1 - a_next_min_ind_weight
            policy{a_now_ind, z_now_ind} = [a_next_min_ind, a_next_min_ind_weight, a_next_max_ind, a_next_max_ind_weight]; % store the index and also the weight
        end
    end

    if ~converged
        error('Value function iteration did not converge');
    end
end