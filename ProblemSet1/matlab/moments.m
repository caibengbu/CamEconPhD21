function [M1, M2, M3, M4] = moments(a,b,r,tau,eta,chi,beta,sigma_z,alpha,A)
    [w, T] = wage_solver(a,b,r,tau,eta,chi,beta,sigma_z,alpha,A);
    [effective_labor, employment, labor_std] = aggregate_labor_supply_quad_approx(w,a,b,r,tau,T,eta,chi,beta,sigma_z);
    unemp = 1 - employment;

    M1 = effective_labor / employment;
    M2 = unemp;
    M3 = unemp * b/(effective_labor * w);
    M4 = labor_std / effective_labor;
end