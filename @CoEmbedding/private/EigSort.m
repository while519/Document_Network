function [ V, D] = EigSort( VB, DB, mode)
% EIGSORT - sort the eigen vector matrix according to the value of eigen
% value
%
% [E, B] = EigSort (A, B, mode)
%
%   A - (DxD) eigenvector matrix
%   B - (DxD) diagonal matrix
%   mode - 'ascend' or 'descend' 
%
% Returns :
%   E - (DxD) The sorted eigen vector matrix
%   B - (Dx1) The sorted sequence of eigenvalue
%
% Description :
%   This fully vectorized m-file computes the
%   the sorted eigen pairs by use:
%               
%               [~,idx]= sort(D,mode);
%
% Example :
%   A=rand(3,3); [P B]=eig(A)
%   [v d]= EigSort(P,B); or [v d]= EigSort(P,B,'ascend');

% Author   : Yu Wu
%            University of Liverpool
%            Electrical Engineering and Electronics
%            Brownlow Hill, Liverpool L69 3GJ
%            yuwu@liv.ac.uk
% Last Rev : Friday, January 20, 2014 (GMT) 14:07 PM
% Tested   : Matlab_R2013b

% Copyright notice: You are free to modify, extend and distribute 
%    this code granted that the author of the original code is 
%    mentioned as the original author of the code.

% Fixed by GTM+0 (1/20/14) to work for xxx
% and to warn for xxx.  Also ensures that 
% output is all xxx, and allows the option of forcing xxx  

if (nargin<3)
    mode='descend';         % default sort in descend manner
end

if (size(VB,2)~=size(DB,1) || size(DB,1)~=size(DB,2))
    error('Matrix for EigSort is either not square matrix or of the same dimensionality');
end

if (strcmp(mode,'ascend') || strcmp(mode,'descend'))
    D= diag(DB);
    [~,idx]= sort(D,mode);
    D=D(idx);
    V=VB(:,idx);
  else 
    error('Sort mode string not match');
end

end

