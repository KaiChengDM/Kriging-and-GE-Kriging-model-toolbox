function R_gek = corrbiquadspline(x,theta,dim,grad)
 
% biquadratic spline correlation matrix of sample points 

%% Initialise

[m n] = size(x);
R_gek = zeros(m*(1+dim));

%% The standard kriging correlation matrix for all samples

mzmax = m*(m-1)/2;          % number of non-zero distances
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
else
  theta = theta(:).';
end

mn = m1*n;   ss = zeros(mn,1);

xi = reshape(abs(d).* repmat(theta,m1,1), mn,1);
 
i1 = find(xi <= 0.4);
i2 = find(0.4 < xi & xi < 1);

if  ~isempty(i1)
  ss(i1) = 1 - 15*xi(i1).^2 + 35*xi(i1).^3 - 195/8*xi(i1).^4;
end
if  ~isempty(i2)
  ss(i2) = 5/3 - 20/3*xi(i2) + 10*xi(i2).^2 - 20/3*xi(i2).^3 + 5/3*xi(i2).^4;
end

ss = reshape(ss,m1,n);
r = prod(ss, 2);

idx = find(r > 0);   o = (1 : m)';   
mu = (10+m)*eps;
R = sparse([ij(idx,1); o], [ij(idx,2); o],[r(idx); ones(m,1)+mu]);   

if strcmp(grad,'off')
    R_gek=[];
    R_gek(1:m,1:m) = R;
    return;
end

%% The first derivative correlation matrices
  
  R_gek(1:m,1:m) = R;

  u = reshape(sign(d) .* repmat(theta,m1,1), mn,1);

  dr = zeros(mn,1);
  if  ~isempty(i1)
    dr(i1) = -u(i1) .* ( -30*xi(i1) + 105*xi(i1).^2 - 195/2*xi(i1).^3);
  end
  if  ~isempty(i2)
    dr(i2) = -u(i2) .* (-20/3 + 20*xi(i2)- 20*xi(i2).^2 + 20/3*xi(i2).^3);
  end

  ii = 1 : m1; dr1=dr;
  for  j = 1 : dim
    sj = ss(:,j);  ss(:,j) = dr1(ii);
    dr1(ii) = prod(ss,2);
    ss(:,j) = sj;   ii = ii + m1;
  end

  dr1 = reshape(dr1,m1,n);

  for i=1:dim
      Sparse_mat = sparse([ij(:,1); o], [ij(:,2); o],[dr1(:,i); zeros(m,1)]); 
%     R_gek(1:m,i*m+1:(i+1)*m)= Sparse_mat-Sparse_mat';
      R_gek(1:m,i*m+1:(i+1)*m)= Sparse_mat-Sparse_mat';
  end

%% The second derivative matrices

for i = 1:dim
      j = i;
      ddr = zeros(mn,1);
      if  ~isempty(i1)
          ddr(i1) = -(-30 + 210*xi(i1) - 585/2*xi(i1).^2).*theta(i)^2;
      end
      if  ~isempty(i2)
          ddr(i2) = -(20 - 40*xi(i2) + 20*xi(i2).^2).*theta(i)^2;
      end
          ii = 1 : m1;
      for  k = 1 : dim
          sj = ss(:,k);  ss(:,k) = ddr(ii);
          ddr(ii) = prod(ss,2);
          ss(:,k) = sj;   ii = ii + m1;
      end
          ddr = reshape(ddr,m1,n);
          Sparse_mat = sparse([ij(:,1); o], [ij(:,2); o],[ddr(:,i); 30*theta(i)^2.*(ones(m,1)+mu)]); 
          R_gek(m*i+1:m*(i+1),m*j+1:m*(j+1)) = Sparse_mat ;  
end


for i = 1:dim
   for j = i+1:dim 
          sj = ss(:,j);  si=ss(:,i); 
          ss(:,j) = dr((j-1)*m1+1:j*m1); ss(:,i) = -dr((i-1)*m1+1:i*m1);
          dr2 = prod(ss,2);
          ss(:,j) = sj;  ss(:,i) = si; 
         
          Sparse_mat = sparse([ij(:,1); o], [ij(:,2); o],[dr2; zeros(m,1)]); 
          R_gek(m*i+1:m*(i+1),m*j+1:m*(j+1)) = Sparse_mat+Sparse_mat'-diag(Sparse_mat);  
   end
end

% Discard computationally small values for numerical stability
% R_gek(abs(R_gek)<eps) = 0;
end


