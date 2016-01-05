function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% number of training examples
m = length(y);

% J_history is useful because, in principal, we should see J(theta) being
% minimized over iterations.
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % The gradient is a vector which points in the direction in the domain
    % that the function decreases/increases in the most. If
    % f(x_1,x_2,...,x_n) is a differentiable function, its gradient is a
    % vector whose components are the partial derviatives of f.
    %
    % In gradient descent with two parameters, theta can be thought of as
    % an (x,y) coordinate, while J(theta) is a z-coordinate. Taking the
    % partial derivative of x and y gives us a 2-vector (the gradient)
    % which points in the direction that J(theta) increases/decreases in
    % the most.
    %
    % This is the why the line of code below looks a lot like the cost
    % function. It is taking the partial derivative of the cost function.
    %
    % For a mathematical explanation of the partial derivatives, see:
    % http://math.stackexchange.com/a/189792/159872.
    %
    % Finally, remember that in this scenario, X is an mx2 matrix where the
    % first column is filled with 1s. This is why we can take both partial
    % derivatives at the same time, even though they have slightly
    % different equations.
    gradient = (1/m) * (X' * ((X * theta) - y));

    % We update theta with the gradient, moderating the step by alpha. If
    % the gradient is positive, theta becomes smaller, i.e. we're going the
    % wrong way. If the gradient is negative, theta becomes bigger, i.e.
    % keep going towards the local minimum.
    theta = theta - alpha * gradient;

    % Save the cost J in every iteration 
    J_history(iter) = computeCost(X, y, theta);
end

end
