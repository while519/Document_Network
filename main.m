% For the document network data, the input is two matrices including the m x n document content matrix W and the m x 2 citation link matrix C
clear classes
clear all
close all
clc

rng(213);        % code initialized
load('./Data/cora.mat');    % the sythetic temporary data file

%    W -- m x n     C -- m x m
lx = categorical(paper_classname);
[idr, idc] = find(C);
C = [];
for ii = 1 : length(idr)
    if idr(ii) < idc(ii)
        C = [C; idr(ii), idc(ii)];
    end
end

% C -- L x 2

% Load the model and run
opt  = {'optimisation', 'ga', 'obj_option', 'linkage_match', 'obj_para', {'knn', 5}};

para_range = {'eta_r',[-1, 10],'eta_c', [-1, 10],'alpha',[0, 0], 'beta',[0,3]};

I    = {'W', W, 'dim',2, 'Linkage', C, 'model_optimisation', opt, 'parameter_range',para_range };

o    = CoEmbedding(I);
o    = training(o);
figure
Plotcluster(o.X', lx);