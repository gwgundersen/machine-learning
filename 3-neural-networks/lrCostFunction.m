function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

hyp = sigmoid(X * theta);

% Regularization term. Importantly, we do not regularize the first
% component, theta(1).
reg_term = (lambda / (2*m)) * sum(theta(2:end).^2);
J = (sum((-y .* log(hyp)) - ((1-y) .* log(1 - hyp))) / m) + reg_term;

reg_term = (lambda / m) * theta(2:end);
% The right-hand expression concatenates 0, for theta(1), with the
% regularization term, which is an (n-1)-vector.
grad = ((X' * (hyp - y))/m) + [0; reg_term];

grad = grad(:);

end
