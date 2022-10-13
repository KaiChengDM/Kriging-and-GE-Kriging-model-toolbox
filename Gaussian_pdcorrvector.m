function r_gek = Gaussian_pdcorrvector(x_pre,model)

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
%  r_gek(1:n1,1:m)=r_g;

%% first order partial derivative correlation matrices

 for i = 1 : dim
    dist = reshape(dx(:,i),m, n1)';
    r_gek((i-1)*n1+1:i*n1,1:m)  = -2*theta(i).*dist.*r_g;
 end

%% second order partial derivative correlation matrices

  for i = 1 : dim
    for j = 1 : dim
        if i == j
           dist = reshape(dx(:,i),m, n1)';
           r_gek((i-1)*n1+1:i*n1,i*m+1:(i+1)*m) = 2*theta(i)*(-2*theta(i)*dist.^2+1).*r_g;
        else
            dist = reshape(dx(:,i),m, n1)'; dist1 = reshape(dx(:,j),m, n1)';
           r_gek((i-1)*n1+1:i*n1,j*m+1:(j+1)*m) = -4*theta(i)*theta(j).*dist.*dist1.*r_g;
        end
    end
  end

end

