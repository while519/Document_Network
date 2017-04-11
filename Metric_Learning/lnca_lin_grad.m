function [F, dF] = lnca_lin_grad(x, X, I, out_dim, lambda)
%%LNCA_LIN_GRAD computes NCA gradient on the specified dataset
%
%   [F, dF] = lnca_lin_grad(x, X, I, out_dim, lambda)
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
dF = dF(:);

end
