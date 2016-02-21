function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

K = size(centroids, 1);
m = size(X, 1);

% distances is an mxK matrix in which each row is a K-vector representing
% the distance from each centroid to the i-th example.
distances = zeros(m, K);

for i = 1:K
    centroid = centroids(i,:);
    % Sum of squares for every sample in the matrix X.
    distances(:,i) = sum(bsxfun(@minus, X, centroid).^2, 2);
end

% Find the index of centroid with the minimum distance to each example.
[discard, idx] = min(distances, [], 2);

end
