function [A ] = lnca( X, I, mode, out_dim, lambda)
%% LNCA - computes the linear projection matrix that best comply with the describe the directed linkage structure
%
% A = lnca(X, I, out_dim, lambda);
%
%   X - (M x N) matrix
%   I - (M x 2) matrix, containing the direct linkages in each row
%   out_dim - scalar or N x out_dim matrix
%   lambda - (scalar)
%
% Returns :
%
%   A - (N x out_dim) projection matrix
%
% Description :
%   This m-file function computes the proposed Linear Neighbourhood Component Analysis. The
% Projection $\mathbf{A}$ is calculated by Gradient descent method:
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
% Last Rev : Monday, April 10, 2017 (GMT) 10:15 AM
% Tested   : Matlab_R2016a
%
% Copyright notice: You are free to modify, extend and distribute
%    this code granted that the author of the original code is
%    mentioned as the original author of the code.
%
% Fixed by GTM+0 (1/17/14) to work for xxx
% and to warn for xxx.  Also ensures that
% output is all xxx, and allows the option of forcing xxx

%%
% initialize
if ~exist('out_dim', 'var') || isempty(out_dim)
    out_dim = 2;    % default output dimensions
end

if ~exist('lambda', 'var') || isempty(lambda)
    lambda = 0;
end

if ~exist('mode', 'var') || isempty(mode)
    mode = 'Leave_One_Out';
end

%%
% some variables required for optimisation
[~, N] = size(X);                           % M samples that are described by N features
learning_rate = 1;                          % learning rate
tol = 10^(-10);

if numel(out_dim) > 1               % if initial projection matrix A is provided
    initial_solution = true;
    A = out_dim;
    out_dim = size(A, 2);
else
    initial_solution = false;
end

if ~ initial_solution
    A = rand(N, out_dim) * 0.01;           %   randomly generate the projection matrix A
end

%% Run the gradient descent algorithms
% calculate the cost and gradient
if strcmp(mode, 'Leave_One_Out')
    [F, dF] = nca_lin_grad(A, X, I, lambda);
elseif strcmp(mode, 'Entropy')
    [F, dF] = tsne_entropy_grad(A, X, I, lambda);
end

iter = 0;   % iteration counter

while true
    iter = iter + 1;
    if ~rem(iter, 2)
        disp(['Iteration of # ' num2str(iter) '; Cost is ' num2str(F) '; Learning rate is ' num2str(learning_rate)]);
    end
    % update A
    A = A - learning_rate * dF;
    if strcmp(mode, 'Leave_One_Out')
        [New_F, dF] = nca_lin_grad(A, X, I, lambda);
    elseif strcmp(mode, 'Entropy')
        [New_F, dF] = tsne_entropy_grad(A, X, I, lambda);
    end
    
    % changes to make for learning rate :: minimizer
    if New_F < F
        learning_rate = 1.01 * learning_rate;
    else
        learning_rate = 0.4 * learning_rate;
    end
    
    % break the loop if no significant changes are made
    if abs(New_F - F) < tol
        disp('average changes in the loss function is less than tol');
        break;
    end
    
    F = New_F;
end

end

function [F, dF] = nca_lin_grad(A, X, I, lambda)
%%NCA_LIN_GRAD Computes NCA gradient on the specified dataset
%
%   [F, dF] = nca_lin_grad(A, X, I, lambda)
%
%   A - (M x out_dim) matrix
%   I - (M x 2) matrix, containing the direct linkages in each row
%   X - (M x N) matrix
%   lambda - (scalar)
%
% Returns :
%
%   F - (scalar) cost function
%   dF - (M x out_dim) gradient

% some variables
cited_index = I(:,1);
citing_index = I(:,2);
nn = size(I, 1);       % number of direct linkages

% Compute the softmax conditional probability
Y = X * A;          % M x out_dim
P = soft_max(Y);
F = 0;
dF = zeros(size(A));

% compute the gradient, summing over each citation
for ii = 1 : nn
    F = F + P(cited_index(ii), citing_index(ii));   %sum cost function
    
    % sum gradient
    xil = bsxfun(@minus, X(cited_index(ii), :), X);
    xilA = bsxfun(@minus, Y(cited_index(ii), :), Y);
    dF = dF + 2 * P(cited_index(ii), citing_index(ii)) * (...
        xil' * bsxfun(@times, P(cited_index(ii), :)', xilA)  ...
        - xil(citing_index(ii), :)' * xilA(citing_index(ii), :));
end

F = -F/nn + lambda .* sum(A(:).^2) ./ numel(A);
dF = -dF/nn + 2 * lambda .* A./numel(A);

end

function [F, dF] = nca_entropy_grad(A, X, I, lambda)
%%NCA_ENTROPY_GRAD Computes NCA gradient on the specified dataset
%
%   [F, dF] = nca_entropy_grad(A, X, I, lambda)
%
%   A - (M x out_dim) matrix
%   I - (M x 2) matrix, containing the direct linkages in each row
%   X - (M x N) matrix
%   lambda - (scalar)
%
% Returns :
%
%   F - (scalar) cost function
%   dF - (M x out_dim) gradient

% some variables
[M, ~] = size(X);
cited_index = I(:,1);
citing_index = I(:,2);

nn = size(I, 1);       % number of direct linkages

% create the prior probability in Q
C = sparse(cited_index, citing_index, ones(nn,1), M, M);
C(1 : M+1 : end) = 0;
Q = bsxfun(@rdivide, C, sum(C, 2));

% Compute the softmax conditional probability
Y = X * A;          % M x out_dim
P = soft_max(Y);
F = 0;
dF = zeros(size(A));

% compute the gradient, summing over each citation
for ii = 1 : nn
    F = F + Q(cited_index(ii), citing_index(ii)) * ...
        log(Q(cited_index(ii), citing_index(ii))/P(cited_index(ii), citing_index(ii)));   %sum cost function
    
    % sum gradient
    xil = bsxfun(@minus, X(cited_index(ii), :), X);
    xilA = bsxfun(@minus, Y(cited_index(ii), :), Y);
    dF = dF - 2 * Q(cited_index(ii), citing_index(ii)) * (...
        xil' * bsxfun(@times, P(cited_index(ii), :)', xilA)  ...
        - xil(citing_index(ii), :)' * xilA(citing_index(ii), :));
end

F = F/nn + lambda .* sum(A(:).^2) ./ numel(A);
dF = dF/nn + 2 * lambda .* A./numel(A);

end


function [F, dF] = tsne_lin_grad(A, X, I, lambda)
%%NCA_LIN_GRAD Computes NCA gradient on the specified dataset
%
%   [F, dF] = nca_lin_grad(A, X, I, lambda)
%
%   A - (M x out_dim) matrix
%   I - (M x 2) matrix, containing the direct linkages in each row
%   X - (M x N) matrix
%   lambda - (scalar)
%
% Returns :
%
%   F - (scalar) cost function
%   dF - (M x out_dim) gradient

% some variables
cited_index = I(:,1);
citing_index = I(:,2);
nn = size(I, 1);       % number of direct linkages

% Compute the softmax conditional probability
Y = X * A;          % M x out_dim
M = size(Y, 1);
sum_Y = sum(Y.^2, 2);       % square row sum: M x 1
num = 1./ (1 + bsxfun(@plus, sum_Y, bsxfun(@minus, sum_Y', 2*(Y*Y')))); % pairwise Euclidean matrix
num(1 : M+1 : end) = 0;
P = bsxfun(@rdivide, num, sum(num, 2));
L = num .* P;
F = 0;
dF = zeros(size(A));

% compute the gradient, summing over each citation
for ii = 1 : nn
    F = F + P(cited_index(ii), citing_index(ii));   %sum cost function
    
    % sum gradient
    xil = bsxfun(@minus, X(cited_index(ii), :), X);
    xilA = bsxfun(@minus, Y(cited_index(ii), :), Y);
    dF = dF + 2 * P(cited_index(ii), citing_index(ii)) * (...
        xil' * bsxfun(@times, L(cited_index(ii), :)', xilA)  ...
        - num(cited_index(ii), citing_index(ii))*xil(citing_index(ii), :)' * xilA(citing_index(ii), :));
end

F = -F/nn + lambda .* sum(A(:).^2) ./ numel(A);
dF = -dF/nn + 2 * lambda .* A./numel(A);

end

function [F, dF] = tsne_entropy_grad(A, X, I, lambda)
%%NCA_ENTROPY_GRAD Computes NCA gradient on the specified dataset
%
%   [F, dF] = nca_entropy_grad(A, X, I, lambda)
%
%   A - (M x out_dim) matrix
%   I - (M x 2) matrix, containing the direct linkages in each row
%   X - (M x N) matrix
%   lambda - (scalar)
%
% Returns :
%
%   F - (scalar) cost function
%   dF - (M x out_dim) gradient

% some variables
[M, ~] = size(X);
cited_index = I(:,1);
citing_index = I(:,2);

nn = size(I, 1);       % number of direct linkages

% create the prior probability in Q
C = sparse(cited_index, citing_index, ones(nn,1), M, M);
C(1 : M+1 : end) = 0;
Q = bsxfun(@rdivide, C, sum(C, 2));

% Compute the softmax conditional probability
Y = X * A;          % M x out_dim
M = size(Y, 1);
sum_Y = sum(Y.^2, 2);       % square row sum: M x 1
num = 1./ (1 + bsxfun(@plus, sum_Y, bsxfun(@minus, sum_Y', 2*(Y*Y')))); % pairwise Euclidean matrix
num(1 : M+1 : end) = 0;
P = bsxfun(@rdivide, num, sum(num, 2));
L = num .* P;
F = 0;
dF = zeros(size(A));

% compute the gradient, summing over each citation
for ii = 1 : nn
    F = F + Q(cited_index(ii), citing_index(ii)) * ...
        log(Q(cited_index(ii), citing_index(ii))/P(cited_index(ii), citing_index(ii)));   %sum cost function
    
    % sum gradient
    xil = bsxfun(@minus, X(cited_index(ii), :), X);
    xilA = bsxfun(@minus, Y(cited_index(ii), :), Y);
    dF = dF - 2 * Q(cited_index(ii), citing_index(ii)) * (...
        xil' * bsxfun(@times, L(cited_index(ii), :)', xilA)  ...
        - num(cited_index(ii), citing_index(ii))*xil(citing_index(ii), :)' * xilA(citing_index(ii), :));
end

F = F/nn + lambda .* sum(A(:).^2) ./ numel(A);
dF = dF/nn + 2 * lambda .* A./numel(A);

end