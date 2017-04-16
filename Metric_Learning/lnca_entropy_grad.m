function [F, dF] = lnca_entropy_grad(x, X, I, out_dim, lambda)
%%LNCA_ENTROPY_GRAD computes NCA gradient for the entropy cost function
%
%   [F, dF] = lnca_entropy_grad(A, X, I, lambda)
%
%   x - (M*out_dim x 1) vector
%   I - (M x 2) matrix, containing the direct linkages in each row
%   X - (M x N) matrix
%   lambda - (scalar)
%
% Returns :
%
%   F - (scalar) cost function
%   dF - (M x out_dim) gradient

if ~exist('lambda', 'var') || isempty(lambda)
    lambda = 0;
end

A = reshape(x, [numel(x) / out_dim out_dim]);

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
dF = dF(:);

end

