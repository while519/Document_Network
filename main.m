% For the document network data, the input is two matrices including the m x n document content matrix W and the m x 2 citation link matrix C
clear classes
clear all
close all
clc

rng(213);        % code initialized
load('tmp.mat');    % the sythetic temporary data file
figure
Plotcluster(X', lx, Y', ly);

%    W -- m x n     C -- m x 2
%    X, Y, lx, ly
% Load the model and run
opt  = {'optimisation', 'ga', 'obj_option', 'shared_knn', 'obj_para', {'knn', 5}};

para_range = {'eta_r',[-1, 10],'eta_c', [-1, 10],'alpha',[0, 3], 'beta',[0,3]};

I    = {'W', W, 'dim',2, 'Linkage', C, 'model_optimisation', opt, 'parameter_range',para_range };

o    = CoEmbedding(I);
o    = training(o);
figure
Plotcluster(o.X', lx, o.Y', ly);