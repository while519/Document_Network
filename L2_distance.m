function d = L2_distance(a, b, mod)
% L2_DISTANCE - computes Euclidean distance matrix
%
% E= L2_distance(A,B)
%
%   A - (DxM) matrix
%   B - (DxN) matrix
%
% Returns :
%   E - (MxN) Euclidean distances between vectors in A and B
%
% Description :
%   This fully vectorized m-file computes the
%   Euclidean distance between two vectors by:
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

if (nargin < 2)
    error('Not enough input arguments');
end

if (nargin<3)
    mod='default';
end
if(size(a,1) ~= size(b,1))
    error('A and B should be of the same dimensionality');
end

if ~(isreal(a)*isreal(b))
    disp('Warning: running L2.distance.m with imaginary numbers. Results may be off.');
end

if (size(a,1)==1)
    a= [a; zeros(1, size(a,2))];
    b= [b; zeros(1, size(b,2))];
end
  
switch mod
    
    case 'default'

ab=a'*b; aa=sum(a.*a); bb=sum(b.*b);
d=sqrt(repmat(aa',[1 size(b,2)])+repmat(bb, [size(a,2) 1])-2*ab);

    case 'knn'
        
ab=a'*b; aa=sum(a.*a); bb=sum(b.*b);
d=repmat(aa',[1 size(b,2)])+repmat(bb, [size(a,2) 1])-2*ab;

end

if(isequal(a,b))
    for i=1:size(a,2)
        d(i,i)=0;
    end
end

end

% Fixed by GTM+0 (1/21/14) to force the diagonals to be zero if the input
% vectors are the same

% Fixed by GTM+0 (1/30/14) to exclude the sqrt calculation in knn

