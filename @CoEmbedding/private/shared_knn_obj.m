function [score] = shared_knn_obj(C, X, Y, obj_option, obj_para)

for ii = 1 : 2 : length(obj_para)-1
    switch lower(obj_para{ii})
        case 'knn'
            if isscalar(obj_para{ii+1})
                knn = obj_para{ii+1};
            end
        otherwise
            error( [ 'unknown input argument: ' obj_para{ii} ] );
    end
end

switch lower(obj_option)
    
    case 'shared_knn'
        positive_score = 0;
        [knn_index_in_Y, ~] = knnsearch(Y, X, 'k', knn);
        
        for ii = 1 : size(C,1)
            positive_score = positive_score + ~isempty(intersect(knn_index_in_Y(C(ii,1), :), knn_index_in_Y(C(ii,2),:))); 
        end
        nb_utilized_knn = length(unique(knn_index_in_Y(:)));
        score = - positive_score/size(C,1) - nb_utilized_knn/size(Y,1) ;
     
    case 'negative_shared_knn'
        positive_score = 0;
        negative_score = 0;
        [knn_index_in_Y, ~] = knnsearch(Y, X, 'k', knn);
        
        for ii = 1 : size(C,1)
            positive_score = positive_score + ~isempty(intersect(knn_index_in_Y(C(ii,1), :), knn_index_in_Y(C(ii,2),:))); 
            negative_score = negative_score + ~isempty(intersect(knn_index_in_Y(C(ii,1), :), knn_index_in_Y(randi(C(ii,1)-1),:))); 
        end
        score = (negative_score - positive_score)/size(C,1) ;
        
    otherwise
        error( [ 'unknown optimization objective function: ' obj_option ] );
end


end