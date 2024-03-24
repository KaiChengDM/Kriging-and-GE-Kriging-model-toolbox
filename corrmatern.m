function R_gek = corrmatern(x,theta,dim,grad)
   
% Matern 5/2 correlation matrix between sample points

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

td = d.* repmat(theta(:).',m1,1);

td = sqrt(sum(td.^2,2));

r = exp(-sqrt(5)*sum(td, 2)).*(1 + sqrt(5)*td + 5/3*td.^2);

idx = find(r > 0);   o = (1 : m)';   
mu = (10+m)*eps;
R = sparse([ij(idx,1); o], [ij(idx,2); o], ...
[r(idx); ones(m,1)+mu]);   

if strcmp(grad,'off')
    R_gek=[];
    R_gek(1:m,1:m) = R;
    return;
end

end



