function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% number of training examples
m = length(y); 

% Compute cost.
hyp = X * theta;
err = hyp - y;
err_sqr = err.^2;
J_temp =  (1/(2*m)) * sum(err_sqr);
reg_term = (lambda/(2*m)) * sum(theta(2:end).^2);
J = J_temp + reg_term;

% Compute gradient.
grad_temp = (X' * (hyp - y)) / m;
reg_term = (lambda / m) * theta(2:end);
grad = grad_temp + [0; reg_term];
grad = grad(:);

end
