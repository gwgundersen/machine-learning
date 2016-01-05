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
        
    % vector of predictions.
    hyp = X * theta;

    % difference between the predicted and actual values.
    err = hyp - y;

    % In gradient descent with two parameters, the gradient is a 2-vector
    % whose components are the two partial derivatives of theta. We can
    % think of theta as an (x,y) coordinate, while J(theta) is the
    % z-coordinate. Taking the partial derivative of each component gives
    % us a vector that points in the direction of greatest/least slope.
    %
    % For a mathematical treatment, see:
    % https://en.wikipedia.org/wiki/Gradient
    % http://math.stackexchange.com/a/189792/159872.
    gradient = (1/m) * (X' * err);

    % We update theta with gradient. If the gradient is positive, theta
    % becomes smaller, i.e. we're going the wrong way. If the gradient is
    % negative, theta becomes bigger, i.e. keep going towards the local
    % minimum.
    %
    % Multiply by alpha to moderate the step.
    theta = theta - alpha * gradient;

    % ============================================================

    % Save the cost J in every iteration 
    J = computeCost(X, y, theta);
    J_history(iter) = J;
end

end
