function [Likelihood Likelihood1 Likelihood2] = GEKriging_Likelihood(x,y,grad_y,corr_fun,theta)   

% Training a gradient-enhanced Kriging model

%%  Preparation

[m n] = size(x);             % number of design sites and their dimension     

dim = n;   % Determine the dimension of active subspace 

mn = m*(dim+1); 

lb_input = min(x);   ub_input =  max(x); 
u = (x-repmat(lb_input,m,1))./(repmat(ub_input,m,1)-repmat(lb_input,m,1));   % Normalization of input data

mean_output = mean(y); std_output = std(y);
y = (y-repmat(mean_output,m,1))./repmat(std_output,m,1);  % Normalization of output data
grad_f = grad_y(:,1:dim).*(ub_input(1:dim)-lb_input(1:dim))/std_output; % Retained partial derivative samples

f      = [ones(m,1); zeros(m*dim,1)];   
grad_d = reshape(grad_f(:,1:dim),dim*m,1); 
yt     = [y; grad_d];      

input_bound = [lb_input; ub_input];  % Input bound
output_moment = [mean_output; std_output]; % Output moment

model.input_bound = input_bound;
model.output_moment = output_moment;
model.tran_input = u;
model.output = y;
model.orig_grad_output = grad_y;
model.tran_grad_output = grad_f;

model.orig_input = x;
model.input_dim = n;
model.sample_size = m;
model.dim = dim;
model.corr_fun = corr_fun;

% %% Likelihood function
% 
%  reduced_corrmat = Corrmat_chol(model,theta);    %  Matrix inversion with cholesky
%     
%  [tran_mat{1} rd(1)] = chol(reduced_corrmat(1:m,1:m));
%  
%  det(1) = prod(diag(tran_mat{1}).^(2/(m+m*dim)));
%  
%  f = ones(m,1);  yt = y;
%  
%  beta0 = f'*(tran_mat{1}\(tran_mat{1}'\yt))/(f'*(tran_mat{1}\(tran_mat{1}'\f)));
% 
%  sig(1) = (yt-beta0*f)'*(tran_mat{1}\(tran_mat{1}'\(yt-beta0*f)))/(m+m*dim);
%  
%   
%  for i = 1 : dim
%        
%    [tran_mat{i+1} rd(i+1)] = chol(reduced_corrmat(m*i+1:(i+1)*m,m*i+1:(i+1)*m));
%     
%    det(i+1) = prod(diag(tran_mat{i+1}).^(2/(m+m*dim))); 
%    
%    grad_d = grad_f(:,dim); yt = grad_d;
%    
%    sig(i+1) = yt'*(tran_mat{i+1}\(tran_mat{i+1}'\yt))/(m+m*dim);
% 
%  end 
%  
%   sigma2 = sum (sig);
%   
%   Likelihood = log(sigma2*prod(det));  % likelihood function of weighted krging model
% 
%   model.upper_mat = tran_mat;

%% Full likelihood function
 
 corrmat  = feval(corr_fun,u,theta,dim,'on');
 
%  condnum = cond(corrmat,2);

 [upper_mat rd] = chol(corrmat);
            
 beta0 = f'*(upper_mat\(upper_mat'\yt))/sum((upper_mat\f).^2);
  
 sigma2 = sum((upper_mat'\(yt-beta0*f)).^2)/mn;

%   detR = prod(diag(upper_mat).^(2/mn));  
%   Likelihood = sigma2*detR;  % likelihood function of weighted krging model
%   Likelihood1 = sigma2;  % likelihood function of weighted krging model
%   Likelihood2 = detR;

%  detR = prod(diag(upper_mat).^2);
%  Likelihood1 = mn*log(sigma2);  % likelihood function of weighted krging model
%  Likelihood2 = log(detR);
%  Likelihood = Likelihood1+Likelihood2;  % likelihood function of weighted krging model

  Likelihood1 = mn*log(sigma2);  % likelihood function of weighted krging model
  Likelihood2 = 2*sum(log(diag(upper_mat)));
  Likelihood = Likelihood1+Likelihood2;  % likelihood function of weighted krging model

end   

