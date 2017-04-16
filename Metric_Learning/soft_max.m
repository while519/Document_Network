function [ P ] = soft_max(Y)
%% SOFT_MAX - computes the softmax probability constructed from the embedded points in Y
%
% P = soft_max(Y);
%
%   Y - (M x k) matrix
%
% Returns :
%
%   P - (M x M) conditional probability matrix, row sum is one
%
% Description :
%   This m-file function computes the conditional probability from the
%   embedding ponits \mathbf{Y}:
%               
%                   \begin{equation} \label{Eq:NCA_obj}
%                        p_{i \rightarrow j} =   \frac{\mathrm{exp}(- \| 
%                        \bm{x}_i \mathbf{A} - \bm{x}_j \mathbf{A} \|^2)}
%                        {\sum_{k \neq i } \mathrm{exp}(-\| \bm{x}_i \mathbf{A} - 
%                        \bm{x}_k \mathbf{A} \|^2)}
%                   \end{equation}
%
% Example : N/A

%%
%
% Author   : Yu Wu
%            University of Liverpool
%            Electrical Engineering and Electronics
%            Brownlow Hill, Liverpool L69 3GJ
%            yuwu@liv.ac.uk
% Last Rev : Monday, April 10, 2017 (GMT) 11:09 AM
% Tested   : Matlab_R2016a
%
% Copyright notice: You are free to modify, extend and distribute 
%    this code granted that the author of the original code is 
%    mentioned as the original author of the code.
%
% Fixed by GTM+0 (1/17/14) to work for xxx
% and to warn for xxx.  Also ensures that 
% output is all xxx, and allows the option of forcing xxx  

M = size(Y, 1);

sum_Y = sum(Y.^2, 2);       % square row sum: M x 1
P = exp(- bsxfun(@plus, sum_Y, bsxfun(@minus, sum_Y', 2*(Y*Y')))); % pairwise Euclidean matrix
P(1 : M+1 : end) = 0;
P = bsxfun(@rdivide, P, sum(P, 2));
%P = max(P, eps);

end

