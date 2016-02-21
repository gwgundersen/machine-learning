function [U, S] = pca(X)
%PCA Run principal component analysis on the dataset X
%   [U, S, X] = pca(X) computes eigenvectors of the covariance matrix of X
%   Returns the eigenvectors U, the eigenvalues (on diagonal) in S
%

% Useful values
[m, n] = size(X);

% Compute the covariance matrix Sigma.
Sigma = (X'*X)/m;

% svd() will compute the principal components from the covariance matrix.
[U, S, V] = svd(Sigma);

end
