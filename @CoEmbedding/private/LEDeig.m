function [ Xq, Yq, time, para, best, ga_out, exitflag, svds] = LEDeig( W, C, dim, options, para_range, display )
%%
% LEDFeig - Runing the LEDF model using eigenvalue decomposition 
%
% [obj.X, obj.Y, obj.cputime, obj.model, obj.best, obj.ga_out]...
%            = LEDeig(obj.W, obj.C, obj.dim, obj.optimisation_option, obj.para_range, obj.display);
%
%           obj.W       -   (m x n) input relational matrix
%           obj.C       -   (m x m) linkage matrix
%           obj.dim     -   (scalar) targeted embedded dimensionality
%           obj.optimisation_options     -   choose the optimisation method
%           obj.para_range  -   parameters( alpha, beta, eta_r, eta_c) range
%
%  Return:
%           Xq,Yq     -   (M,N x dim) row and column embeddings
%           time      -   running time for the model
%           para      -   (struct) parameters for best matched model
%           best      -   best criterion function( score) value
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

% --------------------assign parameter values------------------------------
for ii = 1 : 2 : length(para_range)-1
    switch lower(para_range{ii})
        case 'eta_r'
            if isscalar(para_range{ii+1})
                eta1_upper = para_range{ii+1};
                eta1_lower = para_range{ii+1};
            else
                eta1_lower = para_range{ii+1}(1);
                eta1_upper = para_range{ii+1}(2);
            end
        case 'eta_c'
            if isscalar(para_range{ii+1})
                eta2_lower = para_range{ii+1};
                eta2_upper = para_range{ii+1};
            else
                eta2_lower = para_range{ii+1}(1);
                eta2_upper = para_range{ii+1}(2);
            end
        case 'alpha'
            if isscalar(para_range{ii+1})
                alpha_lower = para_range{ii+1};
                alpha_upper = para_range{ii+1};
            else
                alpha_lower = para_range{ii+1}(1);
                alpha_upper = para_range{ii+1}(2);
            end
        case 'beta'
            if isscalar(para_range{ii+1})
                beta_lower = para_range{ii+1};
                beta_upper = para_range{ii+1};
            else
                beta_lower = para_range{ii+1}(1);
                beta_upper = para_range{ii+1}(2);
            end
        otherwise
            error( [ 'unknown model variables for LEDeig: ' para_range{ii} ] )
    end
end

% ---------------------optimisation options--------------------------------
for ii = 1 : 2 : length(options)-1
    switch lower(options{ii})
        case {'optimisation', 'optimization'}
            opt = options{ii+1};
        case {'obj_option', 'score'}
            obj_type = options{ii+1};           % shared_knn
        case 'obj_para'
            obj_para = options{ii+1};           % knn value
        otherwise
            error( ['unknown setting for optimisation type ' options{ii}])
    end
end



% _____________________________alpha parameter model_______________________

switch lower(opt)
    case 'ga'
        %clc
        S=sprintf(['It is current running LEDeig' '.....']);
        disp(S);
        if ~isequal(eta1_upper, eta1_lower) || ~isequal(eta2_upper,eta2_lower) || ...
                ~isequal(alpha_lower,alpha_upper) || ~isequal(beta_lower, beta_upper)
            options1 = gaoptimset('PopulationSize',52,'UseParallel', false,...
                'Vectorized', 'on','Display','iter','StallGenLimit', 50, ...
                'PlotFcns',@gaplotbestf...
                );
            time = cputime;
            [x,fval,exitflag, ga_out] = ga(@(x) LEDeig_obj(x, W, C, dim, obj_type, obj_para),4,[],[],[],[],...
                [eta1_lower, eta2_lower, alpha_lower, beta_lower],...
                [eta1_upper, eta2_upper, alpha_upper, beta_upper],...
                [],[],options1);
            time = cputime-time;
            best = fval;
        else
            x = [eta1_lower, eta2_lower, alpha_lower, beta_lower];
            time = 0;
            best = 0;
            exitflag = -100;
            ga_out = [];
        end
        
        para.eta1 = x(1);
        para.eta2 = x(2);
        para.alpha = x(3);
        para.beta = x(4);
        [~,Xq, Yq, svds] = LEDeig_model(x, W, C, dim, obj_type, obj_para);
        
    otherwise
        error( ['unrecognised optimisation method ' opt ])
end
end


