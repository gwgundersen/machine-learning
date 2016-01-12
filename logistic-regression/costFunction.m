function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta

% In the notes, we take the dot product of theta' and x, when both where
% n-vectors. This gave us a single hypothesis value.
%
% Here, we take X * theta, which produces an m-vector where each component
% is a dot product of a row in X and theta.
hyp = sigmoid(X * theta);

% 0.693
J = sum((-y .* log(hyp)) - ((1-y) .* log(1 - hyp))) / m;





% =============================================================

end
