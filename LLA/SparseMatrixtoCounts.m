function [ WS , DS ] = SparseMatrixtoCounts( WD )
%% SPARSEMATRIXTOCOUNTS - convert the term document count matrix into vectors of token
%
% [ WS , DS ] = SparseMatrixtoCounts( WD )
%
%   WD - (W x D) matrix
%
% Returns :
%
%   WS - (1 x N) vector where WS(k) contains the vocabulary index of the
%   kth word token.
%
%   DS - (1 x N) vetor where DS(k) contains the document index of the kth
%   word token.
%
% Description :
%   This m-file function convert the term document count matrix into two
%   vectors where the one indicates the vocabulary index if each word
%   token, another consist of the document index of each word token. Each
%   vector can be thought as a vector representation for the whole corpus.
%
% Example : N/A

%%
%
% Author   : Yu Wu
%            University of Liverpool
%            Electrical Engineering and Electronics
%            Brownlow Hill, Liverpool L69 3GJ
%            yuwu@liv.ac.uk
% Last Rev : Thursday, March 22, 2017 (GMT) 11:41 AM
% Tested   : Matlab_R2016a
%
% Copyright notice: You are free to modify, extend and distribute 
%    this code granted that the author of the original code is 
%    mentioned as the original author of the code.
%
% Fixed by GTM+0 (1/17/14) to work for xxx
% and to warn for xxx.  Also ensures that 
% output is all xxx, and allows the option of forcing xxx  

[ ii , jj , ss ] = find( WD );  % ss contain the occurrence of each word token

ntokens = full( sum( sum( WD )));  % total number of tokens

WS = zeros( 1,ntokens );
DS = zeros( 1,ntokens );

startindex = 0;
for i=1:length( ii )
   nt = ss( i );
   % no serial information is considerd, each vocabulary index is simply repeated 
   % to reveal the counts information
   WS( startindex+1:startindex+nt ) = ii( i );  
   DS( startindex+1:startindex+nt ) = jj( i );
 
   startindex = startindex + nt; 
end


