function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% See https://en.wikipedia.org/wiki/Sigmoid_function.
%
% `gdivide` performs element-wise division, to ensure this function works
% for matrices and vectors.
g = gdivide(1, 1 + exp(-z));

end
