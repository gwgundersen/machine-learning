function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Number of training examples
m = size(X, 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% Generate a2, the learn input parameters for the next layer.
a2 = sigmoid(X * Theta1');

% Add ones to the a2 data matrix.
a2 = [ones(m, 1) a2];

% All predictions, an (m x b) matrix, where b is the number of parameters
% in Theta2, the middle layer.
a2 = sigmoid(a2 * Theta2');

[discard, p] = max(a2, [], 2);

end
