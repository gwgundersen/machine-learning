function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% number of training examples
m = length(y); 

% In the notes, we take the dot product of theta' and x, when both were
% n-vectors. This gave us a single hypothesis value.
%
% Here, we take X * theta, which produces an m-vector where each component
% is a dot product of a row in X and theta.
hyp = sigmoid(X * theta);

J = sum((-y .* log(hyp)) - ((1-y) .* log(1 - hyp))) / m;

% Identical to the gradient for linear regression, but the hypothesis
% function has changed.
grad = (X' * (hyp - y)) / m;

end
