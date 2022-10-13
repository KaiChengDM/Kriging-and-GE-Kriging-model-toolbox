function r_gek = Biquadspline_pdcorrvector1(x_pre,model)

% Biquadspline correlation vector between prediction point and sample points

%% Preparation

dim = model.dim;
m = model.sample_size;
n = model.input_dim;

input = model.tran_input;
n1 = size(x_pre,1);
r_gek = zeros(n1*dim,m*(dim+1));

theta = model.theta;

%% Standard kriging correlation matrix

mx = n1; 
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
% r = prod(ss, 2);
% r_g = reshape(r, m, n1)';

%% first order derivative correlation matrices

  u = reshape(sign(dx) .* repmat(theta,m1,1), mn,1);
  dr = zeros(mn,1);

  if  ~isempty(i1)
    dr(i1) =  u(i1) .* ( -30*xi(i1) + 105*xi(i1).^2 - 195/2*xi(i1).^3);
  end
  if  ~isempty(i2)
    dr(i2) = u(i2) .* (-20/3 + 20*xi(i2)- 20*xi(i2).^2 + 20/3*xi(i2).^3);
  end

  ii = 1 : m1; dr1=dr;

  for  j = 1 : dim
    sj = ss(:,j);  ss(:,j) = dr1(ii);
    dr1(ii) = prod(ss,2);
    ss(:,j) = sj;   ii = ii + m1;
  end

  dr1 = reshape(dr1,m1,n);
  
  for i = 1 : dim
      r_gek((i-1)*n1+1:i*n1,1:m) = reshape(dr1(:,i),m,n1)';
  end
  
%% second derivative correlation matrices

 for i = 1 : dim
    for j = 1 : dim
       if i == j
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
             r_gek((i-1)*n1+1:i*n1,i*m+1:(i+1)*m) = reshape(ddr(:,i),m,n1)';
            
       else  
            sj = ss(:,j);  si = ss(:,i); 
            ss(:,j) = dr((j-1)*m1+1:j*m1); ss(:,i) = -dr((i-1)*m1+1:i*m1);
            dr2 = prod(ss,2);
            ss(:,j) = sj;  ss(:,i) = si; 
            
            dr2 = reshape(dr2,n1,m);    
            r_gek((i-1)*n1+1:i*n1,j*m+1:(j+1)*m) = dr2;    

       end
    end
 end
 

end

