function [P, lambda] = nca( X, C, out_dim)
%% LLA - computes linear projection matrix that best comply with the specified linkage structure 
%
% P = lla(X, C, out_dim);
%
%   X - (M x N) matrix
%   C - (M x M) matrix
%   out_dim - scalar
%
% Returns :
%
%   P - (N x out_dim) projection matrix
%
% Description :
%   This m-file function computes the proposed Linear Linkage Analysis. The
% Projection $\mathbf{P}$ is calculated by:
%               
%               \begin{equation}
%                       \min_{\substack{
%                       \mathbf{P} \in \mathcal{R}^{n \times k},\\
%                       \mathbf{P}^{\top} \mathbf{P} = \mathbf{I}_k}
%                       }{ \tr{(\mathbf{P}^{\top} \mathbf{X}^{\top} \mathbf{L}_c \mathbf{X} \mathbf{P})} }
%               \end{equation}
%
% Example : N/A

%%
%
% Author   : Yu Wu
%            University of Liverpool
%            Electrical Engineering and Electronics
%            Brownlow Hill, Liverpool L69 3GJ
%            yuwu@liv.ac.uk
% Last Rev : Friday, February 10, 2017 (GMT) 10:36 AM
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
C = C | C.';        % C is the linkages matrix to indicate the abscence/presence

if ~exist('out_dim', 'var')
    out_dim = 2;    % default output dimensions
end

%%
% Obtain $\mathbf{L}_c$ and $\mathbf{S}_w$
d = sum(C, 2);
Lc = diag(d) - C;
Sw = X'*Lc*X;
Sw = (Sw + Sw.')/2;

%%
% comput the projection matrix
[P, Lambda] = eig(Sw);
[lambda, I] = sort(diag(Lambda), 'ascend');



P = P(:, I(1: out_dim));
lambda = lambda(I(1: out_dim));
disp('LLA Eigenvalues: ');
disp(lambda)

end

