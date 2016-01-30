function [lambda_vec, error_train, error_val] = ...
    validationCurve(X, y, Xval, yval)
%VALIDATIONCURVE Generate the train and validation errors needed to
%plot a validation curve that we can use to select lambda
%   [lambda_vec, error_train, error_val] = ...
%       VALIDATIONCURVE(X, y, Xval, yval) returns the train
%       and validation errors (in error_train, error_val)
%       for different values of lambda. You are given the training set (X,
%       y) and validation set (Xval, yval).
%

% Selected values of lambda (you should not change this)
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';

error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);

for i = 1:length(lambda_vec)
    
   % For each value of lambda
   %   Learn the parameters theta.
   %   Use theta to compute the cost, both for the training set and the
   %     cross validation set.
   %   Store the errors for plotting.

   lambda = lambda_vec(i);
   theta = trainLinearReg(X, y, lambda);
   
   [J_train_i, discard] = linearRegCostFunction(X, y, theta, 0);
   error_train(i) = J_train_i;
   
   [J_cv_i, discard] = linearRegCostFunction(Xval, yval, theta, 0);
   error_val(i) = J_cv_i;
end










% =========================================================================

end
