function ydata = pca_2d(X, labels, no_dims)
% Perform PCA to embedding data into a space of 2 dimensionality
    if ~exist('labels', 'var')
        labels = [];
    end
    if ~exist('no_dims', 'var') || isempty(no_dims)
        no_dims = 2;
    end

    % Normalized input data
    X = X - min(X(:));
    X = X/max(X(:));
    X = bsxfun(@minus, X, mean(X,1));

    disp('Calculating using PCA...');
    if size(X, 2) < size(X, 1)
        C = X' * X;
    else
        C = (1 / size(X, 1)) * (X * X');
    end
    [M, lambda] = eig(C);
    [lambda, ind] = sort(diag(lambda), 'descend');
    M = M(:,ind(1:no_dims));
    lambda = lambda(1:no_dims);
    if ~(size(X, 2) < size(X, 1))
        M = bsxfun(@times, X' * M, (1 ./ sqrt(size(X, 1) .* lambda))');
    end
    ydata = bsxfun(@minus, X, mean(X, 1)) * M;
    clear M lambda ind
    
    figure
    scatter(ydata(:,1), ydata(:,2), 9, labels, 'filled');
    axis tight
    axis off