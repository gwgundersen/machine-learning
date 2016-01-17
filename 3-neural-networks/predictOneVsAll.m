function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% All predictions, i.e. a prediction for each class (10) for each training
% example: 5000x10 matrix.
%
% X is an (m x n) matrix, and theta is (num_labels x n) matrix, so we need
% to tranpose.
all_ps = sigmoid(X * all_theta');

% all_ps is a (m x n) matrix, e.g. 5000x10. Each row represents the scores
% from each classifier. The max score is the winning classification.
%
% `max` will return the indices that contain the max values, and these
% indices map directly to the predicted value, e.g. the score for the
% number "8" is in 8-th classifier, i.e. in the 8-th index.
[discard, p] = max(all_ps, [], 2);

end
