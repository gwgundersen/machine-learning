function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

% Unregularized cost
pred = X * Theta';
err2 = (pred - Y).^2;
J = sum(err2(R == 1)) / 2;

% Gradients for Theta and X
err = (pred - Y) .* R;
Theta_grad = err' * X;
err = (pred - Y) .* R;
X_grad = err * Theta;

% Regularized cost
Theta_reg = (lambda / 2) * sum(sum(Theta.^2));
X_reg = (lambda / 2) * sum(sum(X.^2));
J = J + X_reg + Theta_reg;

% Regularized gradients
Theta_grad_reg = lambda * Theta;
Theta_grad = Theta_grad + Theta_grad_reg;
X_grad_reg = lambda * X;
X_grad = X_grad + X_grad_reg;

grad = [X_grad(:); Theta_grad(:)];

end
