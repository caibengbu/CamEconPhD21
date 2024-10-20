function v = N_func(a, b, r, tau, T, beta)
    c = (b+a*(1+r*(1-tau)) + T)/(1+beta);
    a_prime = beta*c;
    v = log(c) + beta*log(a_prime);
end