function [score,X,Y,BB ] = LEDeig_model(x,W, C, k,obj_option, obj_para )

% x(1,2)    --- eta_1, eta_2
% x(3)      --- alpha
% x(4)      --- bta
% x(5)      --- r
% BB        --- eigen values



eta1=x(1);
eta2=x(2);
t1=x(3);
bta=x(4:end)';


Dr=sum(W,2);             % row sum diagonal matrix     column vector
Dc=sum(W,1);             % column sum diagonal matrix  row vector


Wx=bsxfun(@times, W ,Dr.^eta1);
Pc=sum(Wx,1);

Wy=bsxfun(@times,W,Dc.^(eta2));
Qr=sum(Wy,2);


%% LED
    T1 = bsxfun(@rdivide, Wy, Qr);
    T2 = bsxfun(@rdivide, Wx, Pc);
    
    
    [U,D]=eigs(T1*T2',k+1);                   % S include the nonnegative eigen values in decresing order
    [ U, B] = EigSort( U, D, 'descend');
    
    
    p= 2:k+1;
    
    BB=B(p);                      % the k second largest eigenvalue vector
    sigma = BB/BB(1);
    
    
    
    %% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Xp = bsxfun(@times,(sigma.^bta)',U(:,p));
    gamma = sqrt(1./(diag(U(:,p)'*bsxfun(@times, Qr, U(:,p)))));
    X = bsxfun(@times, Xp, gamma');
    
    Yp = bsxfun(@rdivide,Wx'*X*t1,Pc');
    Y = bsxfun(@rdivide, Yp, sqrt(BB'));

    
    score = shared_knn_obj(C, real(X), real(Y), obj_option, obj_para);

end

