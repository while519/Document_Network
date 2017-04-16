function [ row_idx_train, column_idx_train, row_idx_test, column_idx_test, nb_training_samples, ...
        nb_testing_samples] = sample_linkages( C, sub_sampling_ratio )
%% SAMPLE_LINKAGES - sampling the positive linkages into Train/Test split 
%
% [ row_idx_train, column_idx_train, row_idx_test, column_idx_test, nb_training_samples, ...
%        nb_testing_samples] = sample_linkages( C, sub_sampling_ratio );
%
%   C - (M x M) matrix
%   sub_sampling_ratio - scalar
%
% Returns :
%
%   row_idx_train, column_idx_train - (scalar)  training set indexes
%   row_idx_test, column_idx_test - (scalar)    testing set indexes
%   nb_training_samples, nb_testing_samples - (scalar)
%
% Description :
%   This m-file function randomly sampling a portion of linkages as training 
% set from the provided linkages matrix $\mathbf{C}$.
%
% Example : N/A

%%
%
% Author   : Yu Wu
%            University of Liverpool
%            Electrical Engineering and Electronics
%            Brownlow Hill, Liverpool L69 3GJ
%            yuwu@liv.ac.uk
% Last Rev : Tuesday, March 21, 2017 (GMT) 16:17 PM
% Tested   : Matlab_R2016a
%
% Copyright notice: You are free to modify, extend and distribute 
%    this code granted that the author of the original code is 
%    mentioned as the original author of the code.
%
% Fixed by GTM+0 (1/17/14) to work for xxx
% and to warn for xxx.  Also ensures that 
% output is all xxx, and allows the option of forcing xxx  

%% Split the dataset into training/testing sets
% 
M = size(C, 1);
total_linkages = M * (M - 1) / 2;

% get the indexes of the linkages
C = triu(C);
C = C - diag(diag(C));
[row_idx, column_idx] = find(C > 0);
nb_positives = length(row_idx);


if ~exist('sub_sampling_ratio', 'var')
    nb_training_samples = nb_positives;
else
    nb_training_samples = floor(sub_sampling_ratio * nb_positives);
    if nb_training_samples > nb_positives
        error('Sampling ratio too high, we use instead the total positives');
    end
end

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

end

