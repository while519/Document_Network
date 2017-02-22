function [score] = LEDeig_obj(x,W, C, k,obj_option, obj_para )

% x(:,1,2)    --- eta_1, eta_2
% x(:,3)      --- alpha
% x(:,4)      --- bta

eta1=x(:,1);
eta2=x(:,2);
t1=x(:,3);
bta=x(:,4:end);

len=size(x,1);

Dr=sum(W,2);             % row sum diagonal matrix              column
Dc=sum(W,1);             % column sum diagonal matrix           row

score=zeros(1,len);

parfor ii=1:len
    
    Wx=bsxfun(@times,W,Dr.^(eta1(ii)));
    Pc=sum(Wx,1);                                               %row
    
    Wy=bsxfun(@times,W,Dc.^(eta2(ii)));
    Qr=sum(Wy,2);                                               %column
    
    %% LED
    T1 = bsxfun(@rdivide, Wy, Qr);
    T2 = bsxfun(@rdivide, Wx, Pc);
    

    [U,D]=eigs(T1*T2',k+1, 'lr');                   % S include the nonnegative eigen values in decresing order
    [ U, B] = EigSort( U, D, 'descend');
    
    
    p= 2:k+1;
    
    BB=B(p)';                      % the k second largest eigenvalue vector
    sigma = BB./BB(1);
    
    
    
    %% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Xp = bsxfun(@times,sigma.^bta(ii,:),U(:,p));
    gamma = sqrt(1./(diag(U(:,p)'*bsxfun(@times, Qr, U(:,p)))));
    X = bsxfun(@times, Xp, gamma');             % X emebedding m x k
    
    Y = bsxfun(@rdivide,Wx'*X*t1(ii),Pc');
    Y = bsxfun(@rdivide, Y, sqrt(BB));          % Y embedding n x k
    
    % score
    score(ii) = shared_knn_obj(C, real(X), real(Y), obj_option, obj_para);
end

end

