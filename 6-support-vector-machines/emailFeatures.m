function x = emailFeatures(word_indices)
%EMAILFEATURES takes in a word_indices vector and produces a feature vector
%from the word indices
%   x = EMAILFEATURES(word_indices) takes in a word_indices vector and 
%   produces a feature vector from the word indices. 

% Total number of words in the dictionary
n = 1899;

% x is a binary vector where x_i = 1 if the word is in the email.
% We just index with word_indices and set to 1, indicating that each word
% at each index is in the email, 0 otherwise.
x = zeros(n, 1);
x(word_indices) = 1;

end
