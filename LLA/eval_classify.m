%%% Evaluation by the classification error 

rng(213);

%%
% normalize the data matrix $\mathbf{Z}$
Z = normc(Z);
lz = categorical(webpage_classnames);

%%
% train a nearest neighbors classifier
Mdl = fitcknn(Z, lz, 'NumNeighbors', 1, 'Standardize', 1, 'Distance','cosine');

% examples of accessing properties of Mdl
Mdl.ClassNames
Mdl.Prior

% resubstitution loss
rloss = resubLoss(Mdl)

% Cross-validated classifier
CVMdl = crossval(Mdl);
kloss = kfoldLoss(CVMdl)
