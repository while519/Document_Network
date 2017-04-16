%% Exemplary script for running the network model

%%
% initialize the script
clear classes
clear all
close all
clc

rng(215);

datasets = {'cora', 'citeseer', 'cornell', 'texas', 'washington', 'wisconsin'};
methods = {'lla', 'nla', 'nlla'};

%% parameters loading
sub_sampling_ratio = 0.9;
out_dim = 30;


%% 
% loading the relevent variables/matrices to the workspace
processing_method_id = 3;
processing_data_id = 2;

% load(['../Data/' datasets{processing_data_id} '.mat'], ...
%     'C', 'X', 'webpage_ids', 'webpage_classnames');

load(['../data/' datasets{processing_data_id} '.mat'], ...
        'C', 'X', 'citing_index', 'cited_index', 'webpage_ids', 'webpage_classnames');
C = C | C.';
C0 = C - diag(diag(C));


%% check the conformity between matrix X and C
%
rank_plot(X, citing_index, cited_index, 'Citing');
rank_plot(X, cited_index, citing_index, 'Cited');


%% Split the dataset into training/testing sets
% 
[ row_idx_train, column_idx_train, row_idx_test, column_idx_test, nb_training_samples, nb_testing_samples] ...
    = sample_linkages( citing_index, cited_index, sub_sampling_ratio);

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


%% Run different model
% 
switch methods{processing_method_id}
    case 'lla'
        disp(['It is running the linear model of lla on the dataset of ' datasets{processing_data_id}]);
        Y = preprocessing(X, 300, 'PCA');        % options: PCA, LDA
        P = lla(Y, C, out_dim);
        Z = Y * P;      % embedded points in $\mathbf{Z}$
    
    case 'nla'
        disp(['It is running the nonlinear model of nla on the data set of ' datasets{processing_data_id}]);
        Z = nla(X, C, out_dim, 'cosine');
        
    case 'nlla'
        disp(['It is running the nonlinear model of nlla on the data set of' datasets(processing_data_id)]);
        Y = preprocessing(X, 500, 'PCA');
        P = nlla(X, Y, C, out_dim, 'cosine');
        Z = X * P;
     
    otherwise 
        error('Specifying unused method');
end

%% Plot the evaluation upon all linkages
%
Z = normc(Z);


% %% Evaluation using classification error rate
% % 
% lz = categorical(webpage_classnames);
% 
% % train a nearest neigbours classifier
% Mdl = fitcknn(Z, lz, 'NumNeighbors', 1, 'Standardize', 1, ...
%     'Distance', 'cosine');
% 
% % examing some of the properties of model
% Mdl.ClassNames
% Mdl.Prior
% 
% % resubstitution loss
% rloss = resubLoss(Mdl);
% 
% % Cross-validated classifier
% CVMdl = crossval(Mdl);
% kloss = kfoldLoss(CVMdl);
% disp(['The resubstitution loss for nearest neighbour classifier: ' num2str(rloss)]);
% disp(['The classification error rate: ' num2str(kloss)]);

%% Rank Evaluation
%
% citing
distance = 'euclidean';

[MR, MRR, hitn ] = rank_evals( X, row_idx_train, column_idx_train, 'cosine');
disp(['Content Only Training set Citing Mean Rank: ' num2str(MR)]);
disp(['Content Only Training set Citing Mean Reciprocal Rank: ' num2str(MRR)]);
disp(['Content Only Training set Citing hit@n: ' num2str(hitn)]);

[MR, MRR, hitn ] = rank_evals( Z, row_idx_train, column_idx_train, distance);
disp(['Training set Citing Mean Rank: ' num2str(MR)]);
disp(['Training set Citing Mean Reciprocal Rank: ' num2str(MRR)]);
disp(['Training set Citing hit@n: ' num2str(hitn)]);

% testing ranks
[MR, MRR, hitn ] = rank_evals( X, row_idx_test, column_idx_test, 'cosine');
disp(['Content Only Testing set Citing Mean Rank: ' num2str(MR)]);
disp(['Content Only Testing set Citing Mean Reciprocal Rank: ' num2str(MRR)]);
disp(['Content Only Testing set Citing hit@n: ' num2str(hitn)]);

[MR, MRR, hitn ] = rank_evals( Z, row_idx_test, column_idx_test, distance);
disp(['Testing set Citing Mean Rank: ' num2str(MR)]);
disp(['Testing set Citing Mean Reciprocal Rank: ' num2str(MRR)]);
disp(['Testing set Citing hit@n: ' num2str(hitn)]);


%%
% cited

[MR, MRR, hitn ] = rank_evals( X, column_idx_train, row_idx_train,  'cosine');
disp(['Content Only Training set Cited Mean Rank: ' num2str(MR)]);
disp(['Content Only Training set Cited Mean Reciprocal Rank: ' num2str(MRR)]);
disp(['Content Only Training set Cited hit@n: ' num2str(hitn)]);

[MR, MRR, hitn ] = rank_evals( Z, column_idx_train, row_idx_train,  distance);
disp(['Training set Cited Mean Rank: ' num2str(MR)]);
disp(['Training set Cited Mean Reciprocal Rank: ' num2str(MRR)]);
disp(['Training set Cited hit@n: ' num2str(hitn)]);

% testing ranks
[MR, MRR, hitn ] = rank_evals( X, column_idx_test, row_idx_test, 'cosine');
disp(['Content Only Testing set Cited Mean Rank: ' num2str(MR)]);
disp(['Content Only Testing set Cited Mean Reciprocal Rank: ' num2str(MRR)]);
disp(['Content Only Testing set Cited hit@n: ' num2str(hitn)]);

[MR, MRR, hitn ] = rank_evals( Z, column_idx_test, row_idx_test,  distance);
disp(['Testing set Cited Mean Rank: ' num2str(MR)]);
disp(['Testing set Cited Mean Reciprocal Rank: ' num2str(MRR)]);
disp(['Testing set Cited hit@n: ' num2str(hitn)]);

close all

