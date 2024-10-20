function [c,n] = intensive_labor_supply(w, z, a, r, tau, T, eta, chi, beta)

    c_init = 1.0;
    stepsize = 0.1;
    c_tmp = c_init;
    converged = false;

    iter = 0;

    while iter < 1000
        RHS = (z*w*(1-tau)/(eta*c_tmp))^chi * (z*w*(1-tau)) + a*(1+r*(1-tau)) + T;
        if RHS < 0
            c_tmp = c_tmp * 0.5;
            continue
        end
        c_new = RHS/(1+beta);
        err = abs(c_new - c_tmp);

        if err < 1e-5
            converged = true;
            break
        else
            % fprintf('err = %.5f\n', err);
            c_tmp = stepsize * c_new + (1-stepsize) * c_tmp;
        end
        iter = iter + 1;
    end
    
    if ~converged
        error("converged failed")
    end

    c = c_tmp;
    n = (z*w*(1-tau)/(c_tmp*eta))^chi;
    % fprintf('converged.\n c = %.5f, z = %.5f, w = %.5f.\n', [c,z,w]);
end


