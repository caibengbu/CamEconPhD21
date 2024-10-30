function stationary = stationary_distribution(P)
    % Compute eigenvalues and eigenvectors of the transpose of P
    [eigvecs, eigvals] = eig(P');
    
    % Extract the real parts of the eigenvalues
    real_eigvals = real(diag(eigvals));
    
    % Find the eigenvector corresponding to the largest real eigenvalue
    [~, max_index] = max(real_eigvals);
    stationary = eigvecs(:, max_index);
    
    % Ensure the eigenvector is real
    assert(all(imag(stationary) < 1e-10), 'Imaginary stationary distribution');
    stationary = real(stationary);
    
    % Normalize to sum to 1
    stationary = stationary / sum(stationary);
end