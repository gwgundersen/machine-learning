function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% hyp is a vector of predictions.
hyp = X * theta;

% err is the difference between the predicted and actual values.
err = hyp - y;

% We square the error to remove the sign while retaining proportions.
err_sqr = err.^2;

% The cost is, roughly speaking, the average of all the errors. While not
% explicitly stated, I believe we divide by 2*m rather than just m because
% the two cancels out when taking the partial derivative of the cost
% function.
%
% You should expect to see a cost of 32.07.
J = sum(err_sqr) / (2*m);

% =========================================================================

end
