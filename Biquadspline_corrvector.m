function [r_gek] = Biquadspline_corrvector(x_pre,model,grad)

% Spline correlation vector between prediction point and sample points

%% Initialise
dim = model.dim;
m = model.sample_size;
n = model.input_dim;

input = model.tran_input;
n1 = size(x_pre,1);
r_gek = zeros(n1,m*(dim+1));

theta = model.theta;

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
else
  theta = theta(:).';
end

mn = m1*n;   ss = zeros(mn,1);
xi = reshape(abs(dx) .* repmat(theta,m1,1), mn,1);

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
r_g=reshape(r, m, n1)';

if strcmp(grad,'off')
    r_gek = [];
    r_gek(1:n1,1:m) = r_g;
    return;
end

%% Derivative correlation matrices

  r_gek(1:n1,1:m)=r_g;
  u = reshape(sign(dx) .* repmat(theta,m1,1), mn,1);
  dr = zeros(mn,1);

  if  ~isempty(i1)
    dr(i1) =  -u(i1) .* ( -30*xi(i1) + 105*xi(i1).^2 - 195/2*xi(i1).^3);
  end
  if  ~isempty(i2)
    dr(i2) = -u(i2) .* (-20/3 + 20*xi(i2)- 20*xi(i2).^2 + 20/3*xi(i2).^3);
  end

  ii = 1 : m1;
  for  j = 1 : dim
    sj = ss(:,j);  ss(:,j) = dr(ii);
    dr(ii) = prod(ss,2);
    ss(:,j) = sj;   ii = ii + m1;
  end

  dr = reshape(dr,m1,n);
  
  for i=1:dim
      r_gek(1:n1,i*m+1:(i+1)*m)= reshape(dr(:,i),m,n1)';
  end

end

