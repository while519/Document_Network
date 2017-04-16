function [ MR, MRR, hitn ] = rank_evals( X, row_idx, column_idx,  distance)
%% RANK_EVALS - evaluate the query pairs according to their ranks among the similarity measure
%
% [ MR, MRR, hitn ] = rank_evals( X, row_idx, column_idx,  distance);
%
%   X - (M x N) matrix
%   row_idx, column_idx - (L x 1) vector contains the paired query indexes
%   distance - string
%
% Returns :
%
%   MR - (scalar) mean rank
%   MRR - (scalar) mean reciprocal rank
%   hitn - (scalar) hit@10
%
% Description : N/A
%               
%
% Example : N/A

%%
%
% Author   : Yu Wu
%            University of Liverpool
%            Electrical Engineering and Electronics
%            Brownlow Hill, Liverpool L69 3GJ
%            yuwu@liv.ac.uk
% Last Rev : Tuesday, March 21, 2017 (GMT) 16:51 PM
% Tested   : Matlab_R2016a
%
% Copyright notice: You are free to modify, extend and distribute 
%    this code granted that the author of the original code is 
%    mentioned as the original author of the code.
%
% Fixed by GTM+0 (1/17/14) to work for xxx
% and to warn for xxx.  Also ensures that 
% output is all xxx, and allows the option of forcing xxx  

if ~exist('distance', 'var')
    distance = 'euclidean';
end

n = 10;
MR = 0;
MRR = 0;
hitn = 0;

DistMat = squareform(pdist(X, distance));  % distance metric
[~, idx] = sort(DistMat, 'ascend');  % sorted ascend for each column

% calculate the rank for all pairs
for jj = 1 : length(row_idx)
    rank = find(idx(:, row_idx(jj)) ==  column_idx(jj)) - 1;        % if row contains the citing index, then this returns the citing rank
    
    if (rank <= 0)
        %disp([row_idx(jj), idx(2, row_idx(jj))]);
        %disp([column_idx(jj), idx(2, column_idx(jj))]);
        %warning('rank of 0 occurs!');
    end
    
    MRR = MRR + 1/rank;
    MR = MR + rank;

    if (rank <= n)
        hitn = hitn + 1;
    end
end

MRR = MRR / ( length(row_idx));
MR = MR / ( length(row_idx));
hitn = 100 * hitn / length(row_idx);

end

