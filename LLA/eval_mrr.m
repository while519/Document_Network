%%% Mean Reciprocal Rank Evaluation
%


%%
% extract the query points
[rowidx, columnidx] = find(C > 0);
MRR = 0;                % scalar to store the rank
MR = 0;

%%
% distance matrix
% the data is in $\mathbf{Z}$
DistMat = squareform(pdist(Z));
[~, idx] = sort(DistMat, 'ascend');             % closness is sorted in ascending order in columns

for ii = 1 : length(rowidx)
    rankr = find(idx(:, rowidx(ii)) == columnidx(ii)) -1;
%    rankc = find(idx(:, columnidx(ii)) == rowidx(ii));     % this is the
%    equavalent counterpaert

    MRR = MRR + 1/rankr;
    MR = MR + rankr;
end

MRR = MRR / length(rowidx);
MR = MR / length(rowidx);
