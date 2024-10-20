function err = loss_function(a,b,r,tau,eta,chi,beta,sigma_z,alpha,A,target,weight)
    [M1, M2, M3, M4] = moments(a,b,r,tau,eta,chi,beta,sigma_z,alpha,A);
    moment_vec = [M1, M2, M3, M4];
    err_vec = moment_vec - target;
    err = err_vec*weight*err_vec';
    fprintf('err     = %.5f, b       = %.5f, eta     = %.5f, sigma_z = %.5f.\n', [err, b, eta, sigma_z])
    fprintf('M1(0.33)= %.5f, M2(0.06)= %.5f, M3(0.25)= %.5f, M4(0.70)= %.5f.\n', [M1, M2, M3, M4])
end

