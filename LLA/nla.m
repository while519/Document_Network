function [ Y ] = nla( X, C, out_dim, distance )
%% NLA - computes nonlinear embedding points that best comply with the Linkage and content structure
%
% Y = nla(X, C, out_dim, distance);
%
%   X - (M x N) matrix
%   C - (M x M) matrix
%   out_dim - scalar
%   distance - character strings
%
% Returns :
%
%   Y - (N x out_dim) embedded points
%
% Description :
%   This m-file function computes the best projection matrix $\mathbf{P}$
% which makes the objective $\mathbf{J}_w$ to be as small as possible while
% maximises $\mathbf{J}_b$. Its solution relays on the Rayleigh quotient:
%
%                   \begin{align}
%                       \mathcal{J}(\mathbf{Y}) &= \frac{\tr{(\mathbf{Y}^{\top}
%                       \mathbf{L}_{d} \mathbf{Y})}}{\tr{(\mathbf{Y}^{\top} \mathbf{L}_{c} \mathbf{Y})}}
%                   \end{align}
%
% Example : N/A

%%
%
% Author   : Yu Wu
%            University of Liverpool
%            Electrical Engineering and Electronics
%            Brownlow Hill, Liverpool L69 3GJ
%            yuwu@liv.ac.uk
% Last Rev : Tuesday, March 14, 2017 (GMT) 15:32 PM
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
C = C | C.';        % C indicates the abscence/presence of the linkages

if ~exist('out_dim', 'var')
    out_dim = 2;    % default output dimensionality
end

if ~exist('distance', 'var')
    distance = 'euclidean';
end

%%
% Obtain $\mathbf{L}_c$ and $\mathbf{L}_d$
dc = sum(C, 2);
Lc = diag(dc) - C;
Lc = (Lc + Lc.')/2;

D = squareform(pdist(X, distance));
C_ = ~C;
D = D.*logical(C_);
%D = D/max(D(:));
dd = sum(D, 2);
Ld = diag(dd) - D;
Ld = (Ld + Ld.')/2;

%%
% deal with the annoying null eigenvalues: $\mathbf{Y} = \mathbf{U} * \mathbf{Z}$, where \mathbf{U} is the range
% space of $\mathbf{L}_c$

% this is a bug in the code that svds only takes the 6 largest singular
% values
% [U, S, ~] = svds(Lc);

[U, S, ~] = svd(full(Lc));
tol = max(size(Lc)) * eps(norm(full(Lc)));

idx = find(diag(S) > tol);

if length(idx) < out_dim
    warning(['solutions can not be found for dimensionality of ' num2str(out_dim) ...
        ', using the maximum embedding dimension instead']);
    out_dim = length(idx);
end

U = U(:, idx);

%%
% compute the embedded ponits in $\mathbf{Y}$
[Z, Lambda] = eigs(U'*Ld*U, U'*Lc*U, out_dim);
[lambda, I] = sort(diag(Lambda), 'descend');

Z = Z(:, I(1 : out_dim));
Y = U * Z;
lambda = lambda(I(1 : out_dim));
disp(lambda);

end

