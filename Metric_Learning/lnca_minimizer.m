function [mappedX, mapping] = lnca_minimizer(X, I, method, out_dim, lambda)
%LNCA_MINIMIZER Performs NCA on the specified dataset
%
%   [mappedX, mapping] = lnca_minimizer(X, I, no_dims, lambda)
%
%       X - (M x N) content matrix
%       I - (nn x 2) each row contains the citation linkages
%       out_dim - (scalar)
%       lambda - (scalar) regularizaion parameter
%
%   Return:
%       mappedX - (M x out_dim) mapped embedding points
%       mapping - (struct)  mapping.M contain the projection matrix
%
% Description :
%   This m-file function perform linear Neighborhood Components Analysis (NCA) on the
% dataset specified through X and citations in I to reduce the data dimensionality
% to out_dim dimensions. If valid_X and valid_I are specified, the
% function uses early stopping based on NN errors.
% The function returns a embedded data in mappedX, as well as th mapping.
%
%                        \begin{equation}
%                             \fdv{F}{\mathbf{A}} = \sum_{(i,j) \in \mathcal{I}}
%                             2 p_{i \rightarrow j} \big[ \sum_{l \neq i} p_{i \rightarrow l}
%                             \bm{x}_{il}^{\top} \bm{x}_{il} - \bm{x}_{ij}^{\top} \bm{x}_{ij}
%                             \big] \mathbf{A}
%                        \end{equation}
%
% Example : N/A

%%
%
% Author   : Yu Wu
%            University of Liverpool
%            Electrical Engineering and Electronics
%            Brownlow Hill, Liverpool L69 3GJ
%            yuwu@liv.ac.uk
% Last Rev : Tuesday, April 11, 2017 (GMT) 15:04 PM
% Tested   : Matlab_R2016a
%
% Copyright notice: You are free to modify, extend and distribute
%    this code granted that the author of the original code is
%    mentioned as the original author of the code.
%
% Fixed by GTM+0 (1/17/14) to work for xxx
% and to warn for xxx.  Also ensures that
% output is all xxx, and allows the option of forcing xxx


if ~exist('out_dim', 'var') || isempty(out_dim)
    out_dim = size(X, 2);
end

if ~exist('lambda', 'var') || isempty(lambda)
    lambda = 0;
end


% Make sure data is zero mean
mapping.mean = mean(X, 1);
X = bsxfun(@minus, X, mapping.mean);

% Initialize some variables
max_iter = 10;
[n, d] = size(X);
batch_size = min(5000, n);
no_batches = ceil(n / batch_size);
max_iter = ceil(max_iter / no_batches);
A = randn(d, out_dim) * .01;


%d = checkgrad('lnca_lin_grad', A(:), 1e-5, X, I, out_dim, lambda);
%fprint(d);

%% Main iteration loop      
% Run NCA minimization using three linesearches
[A, f] = minimize(A(:), method, max_iter, X,  I, out_dim, lambda);
A = reshape(A, [d out_dim]);

% Compute embedding
mapping.M = A;
mappedX = X * mapping.M;
