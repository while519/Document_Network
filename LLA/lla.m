function [P] = lla( X, C, out_dim)
% LLA - computes linear projection matrix that best comply with the specified
% linkage structure 
%
% P = lla(X, C, out_dim);
%
%   X - (M x N) matrix
%   C - (M x M) matrix
%   out_dim - scalar
%
% Returns :
%   P - (N x out_dim) projection matrix
%
% Description :
%   This m-file function computes the Linear Linkage Analysis projection
%   matrix for the document network data
%   Euclidean distance between two vectors by $x^2+e^{\pi i}$ :
%               
%               ||A-B||= sqrt( ||A||^2 + ||B||^2 - 2*<A,B>)
%
% Example :
%   A=rand(200,300); B=rand(200,400);
%   dist=L2_distance(A,B);

% Author   : Yu Wu
%            University of Liverpool
%            Electrical Engineering and Electronics
%            Brownlow Hill, Liverpool L69 3GJ
%            yuwu@liv.ac.uk
% Last Rev : Friday, January 17, 2014 (GMT) 10:03 AM
% Tested   : Matlab_R2013b

% Copyright notice: You are free to modify, extend and distribute 
%    this code granted that the author of the original code is 
%    mentioned as the original author of the code.

% Fixed by GTM+0 (1/17/14) to work for xxx
% and to warn for xxx.  Also ensures that 
% output is all xxx, and allows the option of forcing xxx  


end

