function [w,T] = wage_solver(a,b,r,tau,eta,chi,beta,sigma_z,alpha,A)
    w_tmp = 1.0;
    T_tmp = 0.0;
    iter = 0;
    converged = false;
    stepsize = 0.1;

    while iter < 2000
        [effective_labor_supply, employment, second_moment] = aggregate_labor_supply_quad_approx(w_tmp,a,b,r,tau,T_tmp,eta,chi,beta,sigma_z);

        if effective_labor_supply < 1e-8
            w_tmp = 2.0 * w_tmp;
            continue
        end
        
        w_new = (1-alpha)*A*a^alpha*effective_labor_supply^(-alpha);
        T_new = (w_tmp * effective_labor_supply + a*r) * tau - (1-employment)*b;
        err = abs(w_new - w_tmp) + abs(T_new - T_tmp);
        
        if err < 1e-5
            converged = true;
            break
        else
            % fprintf('err = %.5f\n', err);
            w_tmp = stepsize * w_new + (1-stepsize) * w_tmp;
            T_tmp = stepsize * T_new + (1-stepsize) * T_tmp;
        end
        iter = iter + 1;
    end

    w = w_tmp;
    T = T_tmp;

    if ~converged
        fprintf("it didn't converge using naive iteration, use bisection.\n")
       
        iter = 0;
        
        while iter < 1000
            w_tmp = bisection(a,b,r,tau,T_tmp,eta,chi,beta,sigma_z,alpha,A);
            [effective_labor_supply, employment, second_moment] = aggregate_labor_supply_quad_approx(w_tmp,a,b,r,tau,T_tmp,eta,chi,beta,sigma_z);
            T_new = (w_tmp * effective_labor_supply + a*r) * tau - (1-employment)*b;
            err = abs(T_new - T_tmp);
            if err < 1e-5
                converged = true;
                break
            else
                fprintf('err = %.5f\n', err)
                T_tmp = stepsize * T_new + (1-stepsize) * T_tmp;
            end
            iter = iter + 1;
        end

        if ~converged
            error("wage solver didn't converge")
        end
    end
end


function werr = wage_err(w,a,b,r,tau,T_tmp,eta,chi,beta,sigma_z,alpha,A)
    % close the labor gap
    [effective_labor_supply, employment, second_moment] = aggregate_labor_supply_quad_approx(w,a,b,r,tau,T_tmp,eta,chi,beta,sigma_z);
    labor_demand = ((1-alpha)*A/w)^(1/alpha)*a;
    werr = labor_demand - effective_labor_supply;
end

    
function wstar = bisection(a,b,r,tau,T_tmp,eta,chi,beta,sigma_z,alpha,A)
    f = @(x) wage_err(x,a,b,r,tau,T_tmp,eta,chi,beta,sigma_z,alpha,A);
    f_vectorize = @(x) arrayfun(f, x);
    vw = linspace(0.01,10,1000);
    optcondition = f_vectorize(vw);
    lower = sum(optcondition>0);
    upper = lower+1;
    weight = (optcondition(lower) - 0)/(optcondition(lower)-optcondition(upper));
    wstar = vw(upper) * weight + vw(lower) * (1-weight);
end