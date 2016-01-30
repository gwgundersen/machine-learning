function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%
%   In this function, you will compute the train and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, you might want to do this in larger intervals.
%

% Number of training examples
m = size(X, 1);

for i = 1:m
    
    % We want to train and evaluate our learning algorithm on increasingly
    % large subsets of the training set. This will allow us to plot a curve
    % showing (hopefully) the cross-validation error decrease as the number
    % of samples increases.
    X_i = X(1:i, :);
    y_i = y(1:i);
    theta = trainLinearReg(X_i, y_i, lambda);
    
    % The training error for a dataset is the same as the cost function
    % without the regularization parameter. We just pass in lambda=0 below
    % to ensure no regularization term is added.
    [J_train_i, discard] = linearRegCostFunction(X_i, y_i, theta, 0);
    error_train(i) = J_train_i;
    
    % We perform cross-validation against the *entire* validation set. So
    % what's changing each time? The learned parameters theta.
    [J_cv_i, discard] = linearRegCostFunction(Xval, yval, theta, 0);
    error_val(i) = J_cv_i;
end

end
