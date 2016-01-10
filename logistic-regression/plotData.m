function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% `find()` returns a vector of indices of each nonzero element. In this
% case, 0 is false.
pos = find(y == 1);
neg = find(y == 0);

% Select (x,y) coordinates for pos / neg data points.
pos_x_coords = X(pos);
pos_y_coords = X(pos, 2);

neg_x_coords = X(neg);
neg_y_coords = X(neg, 2);

% Plot coordinates
plot(pos_x_coords, pos_y_coords, 'k+', 'LineWidth', 2, 'MarkerSize', 7);
plot(neg_x_coords, neg_y_coords, 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);

hold off;

end
