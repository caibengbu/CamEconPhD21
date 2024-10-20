function v = W_func(c_equi, n_equi, eta, chi, beta)
    a_prime = beta*c_equi;
    v = log(c_equi) - eta*n_equi^(1+1/chi)/(1+1/chi) + beta*log(a_prime);
end