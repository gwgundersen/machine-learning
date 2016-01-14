function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Number of training examples.
m = length(y); 

% hyp is a vector of predictions.
hyp = X * theta;

% err is the difference between the predicted and actual values.
err = hyp - y;

% We square the error to remove the sign while retaining proportions.
err_sqr = err.^2;

% The cost is, roughly speaking, the average of all the errors. While not
% explicitly stated, I believe we divide by 2*m rather than just m because
% the 2 cancels out when taking the partial derivative of the cost
% function.
%
% You should expect to see a cost of 32.07.
J = sum(err_sqr) / (2*m);

% =========================================================================

end
