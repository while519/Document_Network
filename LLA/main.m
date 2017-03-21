%%% Exemplary script for running the linear linkage model

%%
%   initialize the script
clear classes
clear all
close all
clc

rng(213);

datasets = {'cora', 'citeseer', 'cornell', 'texas', 'washington', 'wisconsin'};
methods = {'lla', 'nla'};

%% 
%   loading the relevent variables/matrices to the workspace
processing_method_id = 2;
processing_data_id = 3;

% load(['../Data/' datasets{processing_data_id} '.mat'], ...
%     'C', 'X', 'webpage_ids', 'webpage_classnames');

load(['../data/' datasets{processing_data_id} '.mat'], ...
        'C', 'X', 'webpage_ids', 'webpage_classnames');
C = C | C.';
C0 = C - diag(diag(C));

%% check the conformity between matrix X and C
%
rank_score(X, C);

%% Split the dataset into training/testing sets
% 
sub_sampling_ratio = 0.01;

M = size(C, 1);
total_linkages = M * (M - 1) / 2;
nb_training_samples = floor(sub_sampling_ratio * total_linkages);

% get the indexes of the linkages
C = triu(C);
C = C - diag(diag(C));
[row_idx, column_idx] = find(C > 0);
nb_positives = length(row_idx);
nb_testing_samples = nb_positives - nb_training_samples;
disp(['known positive linkages: #' num2str(nb_positives) ' ' ...
    num2str(100*nb_positives/total_linkages) '%']);
disp(['training linkages: #' num2str(nb_training_samples) ' ' ...
    num2str(100*nb_training_samples/total_linkages) '%']);
disp(['testing linkages: #' num2str(nb_testing_samples) ' ' ...
    num2str(100*nb_testing_samples/total_linkages) '%']);

% random sampling step
idx_permutation = randperm(length(row_idx));
idx_train = idx_permutation(1 : nb_training_samples);
idx_test = idx_permutation(nb_training_samples + 1 : end);

row_idx_train = row_idx(idx_train);
column_idx_train = column_idx(idx_train);
row_idx_test = row_idx(idx_test);
column_idx_test = column_idx(idx_test);

% convert matrix for training
C = sparse(row_idx_train, column_idx_train, ...
    ones(nb_training_samples, 1), M, M);        
C = C | C.';
C = C - diag(diag(C));
    
%%
%   Visualize the matrix strcuture
figure 
imagesc(X);
title('Cross Group Content Matrix X');

figure
imagesc(C);
title('Citation Linkages Matrix C');

%% Run the linear model
% 
switch methods{processing_method_id}
    case 'lla'
        disp('It is running the linear model of lla');
        Y = preprocessing(X, 100, 'PCA');
        P = lla(Y, C, 10);
        Z = Y * P;      % embedded points in $\mathbf{Z}$
    
    case 'nla'
        disp('It is running the nonlinear model of nla');
        Z = nla(X, C, 10, 'cosine');
end

%% Evaluation using classification error rate
% 
Z = normc(Z);
rank_score(Z, C0);
lz = categorical(webpage_classnames);

% train a nearest neigbours classifier
Mdl = fitcknn(Z, lz, 'NumNeighbors', 1, 'Standardize', 1, ...
    'Distance', 'cosine');

% examing some of the properties of model
Mdl.ClassNames
Mdl.Prior

% resubstitution loss
rloss = resubLoss(Mdl);

% Cross-validated classifier
CVMdl = crossval(Mdl);
kloss = kfoldLoss(CVMdl);
disp(['The resubstitution loss for nearest neighbour classifier: ' num2str(rloss)]);
disp(['The classification error rate: ' num2str(kloss)]);

%% Mean Reciprocal Rank Evaluation
% 
MRR = 0;
MR = 0;

[rowidx, columnidx] = find(C > 0);  % find all training linkages

% calculate the distance matrix
DistMat = squareform(pdist(Z));
[~, idx] = sort(DistMat, 'ascend');  % sorted ascend for each column

for ii = 1 : length(rowidx)
    rank_train = find(idx(:, rowidx(ii)) ==  columnidx(ii)) - 1;
    MRR = MRR + 1/rank_train;
    MR = MR + rank_train;
end

MRR = MRR / length(rowidx);
MR = MR / length(rowidx);
disp(['Training set Mean Rank: ' num2str(MR)]);
disp(['Training set Mean Reciprocal Rank: ' num2str(MRR)]);

% testing ranks
MRR = 0;
MR = 0;

for ii = 1 : length(row_idx_test)
    rank_test = find(idx(:, row_idx_test(ii)) == column_idx_test(ii)) -1;
%    rankc = find(idx(:, columnidx(ii)) == rowidx(ii));     % this is the
%    equavalent counterpart

    MRR = MRR + 1/rank_test;
    MR = MR + rank_test;
end

MRR = MRR / length(rowidx);
MR = MR / length(rowidx);
disp(['Testing set Mean Rank: ' num2str(MR)]);
disp(['Testing set Mean Reciprocal Rank: ' num2str(MRR)]);

close all

