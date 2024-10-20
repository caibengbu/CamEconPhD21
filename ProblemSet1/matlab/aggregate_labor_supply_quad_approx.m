function [eff_labor_supply, emp, eff_labor_supply_std] = aggregate_labor_supply_quad_approx(w, a, b, r, tau, T, eta, chi, beta, sigma_z)
    lb = 0.0;
    ub = logninv(1.0 - 1e-8);

    f_wrapper = @(z) individual_labor_supply(w,z,a,b,r,tau,T,eta,chi,beta);
    f_vectorize = @(z) arrayfun(f_wrapper, z);

    g_wrapper = @(z) extensive_labor_supply(w,z,a,b,r,tau,T,eta,chi,beta);
    g_vectorize = @(z) arrayfun(g_wrapper, z);

    eff_labor_supply = integral(@(z) lognpdf(z,0,sigma_z) .* z .* f_vectorize(z), lb, ub);
    emp = integral(@(z) lognpdf(z,0,sigma_z) .* g_vectorize(z), lb, ub);
    eff_labor_supply_second_moment = integral(@(z) lognpdf(z,0,sigma_z) .* (z .* f_vectorize(z)).^2, lb, ub);
    eff_labor_supply_std = sqrt(eff_labor_supply_second_moment - eff_labor_supply^2);
end

