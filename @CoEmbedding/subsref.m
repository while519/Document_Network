function val = subsref(obj, index)

switch index.type
    case '()'
        switch index.subs{:}
            otherwise error('index out of range')
        end
    case '.'
        switch index.subs
            case 'C'
                val = obj.C  ;
            case 'W'
                val = obj.W;
            case 'X'
                val = obj.X  ;
            case 'Y'
                val = obj.Y  ;
            case 'optimisation_option'
                val = obj.optimisation_option ;
            case 'ini'
                val = obj.ini;
            case 'para_range'
                val = obj.para_range;
            case 'dim'
                val = obj.dim;
            case 'cputime'
                val = obj.cputime;
            case 'model'
                val = obj.model;
            case 'rescale'
                val = obj.rescale;
            case 'exitflag'
                val = obj.exitflag;
            case 'display'
                val =obj.display;
            case 'svds'
                val = obj.svds;
            case 'best'
                val=obj.best;
            case 'ga_out'
                val=obj.ga_out;
            otherwise
                error('invalid field name')
        end
    otherwise
        error('undefined variable')
end

end

%------------------------------------------------