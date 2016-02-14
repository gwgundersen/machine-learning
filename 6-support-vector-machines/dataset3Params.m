function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.


vals = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
% 64 = 8*8 = length(vals) * length(vals)
min_error = Inf;
C = 1;
sigma = 0.3;
for i = 1:length(vals)
    for j = 1:length(vals)
        sigma_ = vals(i);
        C_ = vals(j);
        x1 = X(:,1);
        x2 = X(:,2);
        model = svmTrain(X, y, C_, @(x1, x2) gaussianKernel(x1, x2, sigma_)); 
        predictions = svmPredict(model, Xval);
        
        % Compute prediction error and check if it is the lowest so far.
        error = mean(double(predictions ~= yval));
        if error < min_error
            min_error = error;
            C = C_;
            sigma = sigma_;
        end
    end
end

end
