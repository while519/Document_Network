function [  ] = rank_plot(X, C)
%% RANK_PLOT - access the conformity between content and linkages using different different measure
%
%  rank_plot(X, C);
%
%   X - (M x N) matrix
%   C - (M x M) matrix
%
%
% Description :
%   This m-file function computes the mean rank and mean reciprocal rank score
% of the pared linkages in $\mathbf{C}$ in terms of different distance measure of $\mathbf{X}$:
%
% Example : N/A

%%
%
% Author   : Yu Wu
%            University of Liverpool
%            Electrical Engineering and Electronics
%            Brownlow Hill, Liverpool L69 3GJ
%            yuwu@liv.ac.uk
% Last Rev : Sunday, March 19, 2017 (GMT) 16:03 PM
% Tested   : Matlab_R2016a
%
% Copyright notice: You are free to modify, extend and distribute
%    this code granted that the author of the original code is
%    mentioned as the original author of the code.
%
% Fixed by GTM+0 (1/17/14) to work for xxx
% and to warn for xxx.  Also ensures that
% output is all xxx, and allows the option of forcing xxx

distances = {'euclidean', 'seuclidean', 'cityblock', 'minkowski', 'chebychev', ...
     'cosine', 'correlation', 'spearman', 'hamming', 'jaccard'};

C = triu(C);
C = C - diag(diag(C));
[row_idx, column_idx] = find(C > 0);

MR_list = [];
MRR_list = [];
hitn_list = [];
for ii = 1 : 10
    [ MR, MRR, hitn ] = rank_evals( X, row_idx, column_idx,  distances{ii});
    MR_list = [MR_list; MR];
    MRR_list = [MRR_list; MRR];
    hitn_list = [hitn_list; hitn];
end

figure
bar([MR_list]);
legend({'Mean Rank'});
set(gca, 'XTick', 1:10, 'XTickLabel', distances);

figure
bar([MRR_list]);
legend({'Mean Reciprocal Rank'});
set(gca, 'XTick', 1:10, 'XTickLabel', distances);

figure
bar([hitn_list]);
legend({'Hit@100 Rate'});
set(gca, 'XTick', 1:10, 'XTickLabel', distances);

end

