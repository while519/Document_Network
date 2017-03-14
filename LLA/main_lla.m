%%% Exemplary script for running the linear linkage model

%%
%   initialize the script
clear classes
clear all
close all
clc

rng(213);

datasets = {'cornell', 'texas', 'washington', 'wisconsin'};

%% 
%   loading the relevent variables/matrices to the workspace
processing_data_id = 1;
load(['../Data/' datasets{processing_data_id} '.mat'], ...
    'C', 'X', 'webpage_ids', 'webpage_classnames');

%%
%   Visualize the matrix strcuture
figure 
imagesc(X);
title('Cross Group Content Matrix X');

figure
C = C| C.';
C = C - diag(diag(C));
imagesc(C);
title('Citation Linkages Matrix C');

%%
% Run the linear model
Y = preprocessing(X, 100, 'PCA');
P = lla(Y, C, 10);
Z = Y * P;      % embedded points in $\mathbf{Z}$