%% Exemplary script for running the network model

%%
% initialize the script
clear classes
clear all
close all
clc

rng('default');
rng(215);

datasets = {'cora', 'citeseer', 'cornell', 'texas', 'washington', 'wisconsin'};
% methods = {'lla', 'nla'};

%% parameters loading
sub_sampling_ratio = 0.9;
out_dim = 30;


%% 
% loading the relevent variables/matrices to the workspace
% processing_method_id = 1;
processing_data_id = 2;

% load(['../Data/' datasets{processing_data_id} '.mat'], ...
%     'C', 'X', 'webpage_ids', 'webpage_classnames');

load(['../data/' datasets{processing_data_id} '.mat'], ...
        'C', 'X', 'citing_index', 'cited_index', 'webpage_ids', 'webpage_classnames');
C = C | C.';
C0 = C - diag(diag(C));


%% Code using dirichlet allocation to reveal the content matrix structure
[WS, DS] = SparseMatrixtoCounts(X');

% hyperparameters setting
T = 50;     % number of topics
BETA = 0.01;
ALPHA = 50 / T;
N = 300;    % number of iterations
SEED = 3;   % random seed
OUTPUT = 1; % what output to show(0 = no output; 1 = iterations; 2 = all output)

%%
% this function might need a few minutes to finish
tic
[ WP,DP, Z ] = GibbsSamplerLDA( WS , DS , T , N , ALPHA , BETA , SEED , OUTPUT );
toc
 
save(['lda_' datasets{processing_data_id}], 'WP', 'DP', 'Z', 'ALPHA', 'BETA', 'SEED', 'N');

% visualize the results
figure
imagesc(WP);
title('Word topic matrix structure');

figure
imagesc(DP);
title('Document topic matrix structure');

%%
% check the similarity measurement for performances
DP = full(DP);


%% Rank Evaluation
%
% citing
distance = 'euclidean';

[MR, MRR, hitn ] = rank_evals( X, citing_index, cited_index, 'cosine');
disp(['Content Only Citing Mean Rank: ' num2str(MR)]);
disp(['Content Only Citing Mean Reciprocal Rank: ' num2str(MRR)]);
disp(['Content Only Citing hit@n: ' num2str(hitn)]);

[MR, MRR, hitn ] = rank_evals( DP, citing_index, cited_index, distance);
disp(['Topic matrix Citing Mean Rank: ' num2str(MR)]);
disp(['Topic matrix Citing Mean Reciprocal Rank: ' num2str(MRR)]);
disp(['Topic matrix Citing hit@n: ' num2str(hitn)]);


%%
% cited

[MR, MRR, hitn ] = rank_evals( X, cited_index, citing_index,  'cosine');
disp(['Content Only Cited Mean Rank: ' num2str(MR)]);
disp(['Content Only Cited Mean Reciprocal Rank: ' num2str(MRR)]);
disp(['Content Only Cited hit@n: ' num2str(hitn)]);

[MR, MRR, hitn ] = rank_evals( DP, cited_index, citing_index,  distance);
disp(['Topic matrix Cited Mean Rank: ' num2str(MR)]);
disp(['Topic matrix Cited Mean Reciprocal Rank: ' num2str(MRR)]);
disp(['Topic matrix Cited hit@n: ' num2str(hitn)]);


%% Topic citation analysis
%
DP = bsxfun(@rdivide, DP, sum(DP, 2));
[B, I] = sort(DP, 2, 'descend');
cum_B = cumsum(B, 2);
