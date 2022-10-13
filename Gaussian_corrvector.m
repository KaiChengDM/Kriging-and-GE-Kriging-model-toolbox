function [r_gek] = Gaussian_corrvector(x_pre,model,grad)

% Gaussian correlation vector between prediction point and sample points

%% Initialise

dim=model.dim;
m=model.sample_size;
n=model.input_dim;

input=model.tran_input;
theta=model.theta;

n1=size(x_pre,1);
r_gek = zeros(n1,m*(dim+1));

%% Standard kriging correlation matrix

 mx=n1; 
 dx = zeros(mx*m,n);  kk = 1:m;
 for  k = 1 : mx
      dx(kk,:) = repmat(x_pre(k,:),m,1) - input;
      kk = kk + m;
 end
[m1 n] = size(dx);  % number of differences and dimension of data
if  length(theta) == 1
  theta = repmat(theta,1,n);
elseif  length(theta) ~= n
  error(sprintf('Length of theta must be 1 or %d',n))
end

 td = dx.^2 .* repmat(-theta(:).',m1,1);
 r = exp(sum(td, 2));
 r_g = reshape(r, m, mx)';
 r_gek(1:n1,1:m)=r_g;

 if strcmp(grad,'off')
    r_gek = [];
    r_gek(1:n1,1:m) = r_g;
    return;
 end
%% Derivative correlation matrices

 for i=1:dim
   dist=reshape(dx(:,i),m, mx)';
   r_gek(1:n1,i*m+1:(i+1)*m)=2*theta(i).*dist.*r_g;
 end

end

