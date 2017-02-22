function obj = training( obj )
%%
% TRAIN - CoEmbeddings model training
%
%  Obj = train(Obj)
%
%           obj     -   co-embeddings object
%
%  Return:
%           Obj     -   co-embeddings object
%
%%
%  Author   : Yu Wu
%            University of Liverpool
%            Electrical Engineering and Electronics
%            Brownlow Hill, Liverpool L69 3GJ
%            yuwu@liv.ac.uk
%  Last Rev : Friday, January 17, 2014 (GMT) 10:03 AM
%  Tested   : Matlab_R2014b
%
%%
% Copyright notice: You are free to modify, extend and distribute
%       this code granted that the author of the original code is
%       mentioned as the original author of the code.

% -------------------------------------------------------------------------

[obj.X, obj.Y, obj.cputime, obj.model, obj.best, obj.ga_out, obj.exitflag, obj.svds]...
            = LEDeig(obj.W, obj.C, obj.dim, obj.optimisation_option, obj.para_range, obj.display);

end

