%% Exemplary script for running the network model

%%
% initialize the script
clear classes
clear all
close all
clc

rng(215);

datasets = {'cora', 'citeseer', 'cornell', 'texas', 'washington', 'wisconsin'};
methods = {'lnca_lin_grad', 'lnca_entropy_grad', 'ltsne_lin_grad', 'ltsne_entropy'};

%% parameters loading
out_dim = 2;


%%
% loading the relevent variables/matrices to the workspace
processing_method_id = 3;
processing_data_id = 2;


load(['../data/' datasets{processing_data_id} '.mat']);
C = C | C.';
C0 = C - diag(diag(C));
[cited_index, citing_index] = find(C0);
I = [cited_index, citing_index];
X = W;


%% check the conformity between matrix X and C
%
%rank_plot(X, C0);


%% Run the linear model
%
disp(['running lnca model with ' methods{processing_method_id} ' function']);
Y = preprocessing(X, 50, 'PCA');
[mappedX, mapping] = lnca_minimizer(Y, I, methods{processing_method_id}, out_dim);
% switch methods{processing_method_id}
%     case 'lnca'
%         disp(['It is running the linear neibourghood component analysis model on the dataset of ' datasets{processing_data_id}]);
%         Y = preprocessing(X, 50, 'PCA');
%         A = lnca(Y, I, 'Leave_One_Out', out_dim);
%         Z = Y * A;      % embedded points in $\mathbf{Z}$
%     case 'lnca-entropy'
%         disp(['It is running the linear entropy neibourghood component analysis model on the dataset of ' datasets{processing_data_id}]);
%         Y = preprocessing(X, 50, 'PCA');
%         A = lnca(Y, I, 'Entropy', out_dim);
%         Z = Y * A;      % embedded points in $\mathbf{Z}$
%
%     case 'lnca_minimizer'
%         disp('running lnca model with minimizer function');
%         Y = preprocessing(X, 500, 'PCA');
%         [mappedX, mapping] = lnca_minimizer(Y, I, methods{processing_method_id}, out_dim);
% end

