function ifwork = extensive_labor_supply(w,z,a,b,r,tau,T,eta,chi,beta)
    [c_ifwork, n_ifwork] = intensive_labor_supply(w,z,a,r,tau,T,eta,chi,beta);
    N_val = N_func(a,b,r,tau,T,beta);
    W_val = W_func(c_ifwork,n_ifwork,eta,chi,beta);
    if W_val > N_val
        % n = n_ifwork;
        ifwork = 1.0;
    else
        % n = 0.0;
        ifwork = 0.0;
    end
end