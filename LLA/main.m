%%% Exemplary script for running the linear linkage model

%%
% initialize the script
clear classes
clear all
close all
clc

rng(213);

datasets = {'cora', 'citeseer', 'cornell', 'texas', 'washington', 'wisconsin'};
methods = {'lla', 'nla'};

%% 
% loading the relevent variables/matrices to the workspace
processing_method_id = 2;
processing_data_id = 2;

% load(['../Data/' datasets{processing_data_id} '.mat'], ...
%     'C', 'X', 'webpage_ids', 'webpage_classnames');

load(['../data/' datasets{processing_data_id} '.mat'], ...
        'C', 'X', 'citing_index', 'cited_index', 'webpage_ids', 'webpage_classnames');
C = C | C.';
C0 = C - diag(diag(C));


%% check the conformity between matrix X and C
%
rank_plot(X, C0);

%% Split the dataset into training/testing sets
% 
sub_sampling_ratio = 0.9;
[ row_idx_train, column_idx_train, row_idx_test, column_idx_test, nb_training_samples] ...
    = sample_linkages( C0, sub_sampling_ratio);

% convert matrix for training
C = sparse(row_idx_train, column_idx_train, ...
    ones(nb_training_samples, 1), size(C0, 1), size(C0, 2));        
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
        Z = nla(X, C, 20, 'cosine');
end

%% Plot the evaluation upon all linkages
%
Z = normc(Z);
rank_plot(Z, C0);

%% Evaluation using classification error rate
% 
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
distance = 'euclidean';

[MR, MRR, hitn ] = rank_evals( X, row_idx_train, column_idx_train, distance);
disp(['Content Only Training set Mean Rank: ' num2str(MR)]);
disp(['Content Only Training set Mean Reciprocal Rank: ' num2str(MRR)]);
disp(['Content Only Training set hit@n: ' num2str(hitn)]);

[MR, MRR, hitn ] = rank_evals( Z, row_idx_train, column_idx_train, distance);
disp(['Training set Mean Rank: ' num2str(MR)]);
disp(['Training set Mean Reciprocal Rank: ' num2str(MRR)]);
disp(['Training set hit@n: ' num2str(hitn)]);

% testing ranks
[MR, MRR, hitn ] = rank_evals( X, row_idx_test, column_idx_test, distance);
disp(['Content Only Testing set Mean Rank: ' num2str(MR)]);
disp(['Content Only Testing set Mean Reciprocal Rank: ' num2str(MRR)]);
disp(['Content Only Testing set hit@n: ' num2str(hitn)]);

[MR, MRR, hitn ] = rank_evals( Z, row_idx_test, column_idx_test, distance);
disp(['Testing set Mean Rank: ' num2str(MR)]);
disp(['Testing set Mean Reciprocal Rank: ' num2str(MRR)]);
disp(['Testing set hit@n: ' num2str(hitn)]);

close all

