function R_gek = corrgaussian(x,theta,dim,grad)
   
% Gaussian correlation matrix between sample points

%%  Initialise

[m n] = size(x);
 R_gek = zeros(m*(1+dim));

%% The standard kriging correlation matrix for all samples

mzmax = m*(m-1) / 2;        % number of non-zero distances
ij = zeros(mzmax, 2);       % initialize matrix with indices
d = zeros(mzmax, n);        % initialize matrix with distances
ll = 0;
for k = 1 : m-1
  ll = ll(end) + (1 : m-k);
  ij(ll,:) = [repmat(k, m-k, 1) (k+1 : m)'];        % indices for sparse matrix
  d(ll,:) = repmat(x(k,:), m-k, 1) - x(k+1:m,:);    % differences between points
end

[m1 n] = size(d);  % number of differences and dimension of data
if  length(theta) == 1
  theta = repmat(theta,1,n);
elseif  length(theta) ~= n
  error(sprintf('Length of theta must be 1 or %d',n))
end

td = d.^2 .* repmat(-theta(:).',m1,1);

r = exp(sum(td, 2));

idx = find(r > 0);   o = (1 : m)';   
mu = (10+m)*eps;
R = sparse([ij(idx,1); o], [ij(idx,2); o], ...
[r(idx); ones(m,1)+mu]);   

if strcmp(grad,'off')
    R_gek=[];
    R_gek(1:m,1:m) = R;
    return;
end

%% The first derivative correlation matrices

R_gek(1:m,1:m) = R;

for i = 1 : dim
    Sparse_mat = sparse([ij(idx,1); o], [ij(idx,2); o],[2*theta(i)*d(:,i).*r(idx); zeros(m,1)]); 
    R_gek(1:m,m*i+1:m*(i+1)) = Sparse_mat-Sparse_mat';  
end

%% The second derivative matrices

for i = 1 : dim
     j = i;
     Sparse_mat = sparse([ij(idx,1); o], [ij(idx,2); o],[2*theta(i)*(-2*theta(i)*d(:,i).^2+1).*r(idx); 2*theta(i).*(ones(m,1)+mu)]); 
     R_gek(m*i+1:m*(i+1),m*j+1:m*(j+1)) = Sparse_mat;  
end

for i = 1:dim
   for j = i+1:dim     
      Sparse_mat = sparse([ij(idx,1); o], [ij(idx,2); o],[-4*theta(i)*theta(j).*d(:,i).*d(:,j).*r(idx); zeros(m,1)]); 
      R_gek(m*i+1:m*(i+1),m*j+1:m*(j+1)) = Sparse_mat+Sparse_mat'-diag(Sparse_mat);  
   end
end

% Discard computationally small values for numerical stability
R_gek(abs(R_gek)<eps) = 0;
end



