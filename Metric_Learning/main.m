%% Exemplary script for running the network model

%%
% initialize the script
clear classes
clear all
close all
clc

rng(215);

datasets = {'cora', 'citeseer', 'cornell', 'texas', 'washington', 'wisconsin'};
methods = {'lnca_lin_grad', 'lnca_entropy_grad', 'ltsne_lin_grad', 'ltsne_entropy_grad'};

%% parameters loading
out_dim = 2;


%%
% loading the relevent variables/matrices to the workspace
processing_method_id = 4;
processing_data_id = 1;
k = 10;


load(['../data/' datasets{processing_data_id} '.mat']);
% X I webpage_classnames webpage_ids


%% 10-fold randomly split the data into training/testing sets
%
CVO = cvpartition(size(X,1), 'KFold', k);


%% Run the linear model
%
MR = zeros(CVO.NumTestSets,1);
disp(['running lnca model with ' methods{processing_method_id} ' function']);
C0 = sparse(cited_index, citing_index, cited_index, size(X,1), size(X,1));
Y = preprocessing(X, 50, 'PCA');
for i = 1:CVO.NumTestSets
    % training/testing indexes
    trIdx = CVO.training(i);
    teIdx = CVO.test(i);
    
    % only sustaining the training linakages, notice that it is not
    % necessary to remove testing documents in X as they will have no
    % effect in the training process
    tr_indexes = find(trIdx);
    Lia = ismember(I, tr_indexes);
    trI_indicator = logical(prod(Lia, 2));
    trI = I(trI_indicator, :);
    teI = I(~trI_indicator, :);
    
    % mappedX contains the embedded points, mapping.M contains the
    % transformation matrix
    [mappedX, mapping] = lnca_minimizer(Y, trI, methods{processing_method_id}, out_dim);
    
    % simply calculate the rank of each pair of links through euclidean
    % distances
    MR(i) = rank_evals(mappedX, teI(:,1), teI(:,2));
end
cvMR = sum(MR)/sum(CVO.TestSize);

