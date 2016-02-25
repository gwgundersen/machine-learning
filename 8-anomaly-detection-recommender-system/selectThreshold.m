function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    
    cvPredictions = pval < epsilon;
    
    % true and false positives; false negative
    tp = sum(cvPredictions == 1 & yval == 1);
    fp = sum(cvPredictions == 1 & yval == 0);
    fn = sum(cvPredictions == 0 & yval == 1);
    
    % F1 score, which is a balanced metric for how well both the precision
    % and recall did. Note that if precision or recall is very low or 0,
    % the F1 score will be (cf. with just taking an average of the two
    % metrics).
    precision = tp / (tp + fp);
    recall = tp / (tp + fn);
    F1 = (2 * precision * recall) / (precision + recall);
    
    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
