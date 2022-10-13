clc;  clear;

n = 2;

g = @(x) x(:,1).^2./4000 + x(:,2).^2./4000 -cos(x(:,1)./1).*cos(x(:,2)./sqrt(2))+1;

Pd{1} = @(x)x(:,1)./2000 +sin(x(:,1)./1).*cos(x(:,2)./sqrt(2));
Pd{2} = @(x)x(:,2)./2000 +cos(x(:,1)./1).*sin(x(:,2)./sqrt(2))./sqrt(2);

% g = @(x)-(10-(0.5.*x(:,1).^2-5.*cos(2*pi*x(:,1)))-(0.6.*x(:,2).^2-5.*cos(3*pi*x(:,2)))); 
% 
% Pd{1} = @(x)-(-1.*x(:,1)-10*pi.*sin(2*pi*x(:,1)));
% Pd{2} = @(x)-(-1.2.*x(:,2)-15*pi.*sin(3*pi*x(:,2)));

Samplesize = 50 : 50: 250;
sig = ones(1,n); mu = zeros(1,n);

for ii = 1 : 10
    u{ii} = normcdf(lhsnorm(mu,diag(sig.^2),10^4));
for k = 1 : 5
 
 x = []; y = []; grad_y = [];

 lb = -20.*ones(1,n);  ub = 20.*ones(1,n); N = Samplesize(k); N1 = 3000; 

%  pp = sobolset(n,'Skip',3); u=net(pp,N);  
%  pp1 = sobolset(n,'Skip',100,'Leap',N1); u1=net(pp1,N1); 

 for i=1:n
    x(:,i) = u{ii}(1:N,i)*(ub(i)-lb(i))+lb(i);
    xtest(:,i) = u{ii}(N+1:N+N1,i)*(ub(i)-lb(i))+lb(i);
 end

y = g(x); y1 = g(xtest);
 
for i = 1:N
   Par =[];
  for j = 1:n
    Par_output(i) = Pd{j}(x(i,:));
    Par = [Par Par_output(i)];
  end
  grad_y(i,:) = Par;
end

%% GE-Kriging
% 
hyperpar.theta = 0.1.*ones(1,n); 
hyperpar.lb = 5*10^-4.*ones(1,n);
hyperpar.ub = 10*ones(1,n);
hyperpar.corr_fun = 'corrbiquadspline';
hyperpar.opt_algorithm = 'Hooke-Jeeves';
hyperpar.multistarts = 10;

inputpar.lb = lb;
inputpar.ub = ub;
inputpar.x = x;
inputpar.y = y;
inputpar.grad = grad_y;

t1=clock;
 GEKriging_Model = GEKriging_fit(inputpar,hyperpar);
t2=clock;

Time(ii,k) = etime(t2,t1)
[Mean Variance] = GEKriging_predictor(xtest,GEKriging_Model);
MSE(ii,k)  = mean((Mean-y1).^2)/var(y1)

%% Kriging

% hyperpar.theta = 0.1.*ones(1,n); 
% hyperpar.lb = 10^-3.*ones(1,n);
% hyperpar.ub = 5*ones(1,n);
% hyperpar.corr_fun = 'corrbiquadspline';
% hyperpar.opt_algorithm = 'Hooke-Jeeves';
% hyperpar.multistarts = 10;
% 
% inputpar.x = x;
% inputpar.y = y;
% 
% t1=clock;
%   Kriging_Model = Kriging_fit(inputpar,hyperpar);
% t2=clock;
% 
% Time1(ii,k) = etime(t2,t1)
% [Mean Variance] = Kriging_predictor(xtest,Kriging_Model);
% MSE1(ii,k)  = mean((Mean-y1).^2)/var(y1)

%% Sliced GE-Kriging

hyperpar.beta = [0.5 0.5 10^-2];
hyperpar.lb   = [5*10^-4 0.2 5*10^-4]; 
hyperpar.ub   = [5 1 5];

hyperpar.corr_fun = 'corrbiquadspline';
%hyperpar.corr_fun = 'corrgaussian';
% hyperpar.opt_algorithm = 'Fmincon';
hyperpar.opt_algorithm = 'Hooke-Jeeves';
hyperpar.multistarts = 10;
inputpar.threshold = 1;
inputpar.x = x;
inputpar.y = y;
inputpar.grad = grad_y;
inputpar.snum = 10;

t1 = clock;
  GEKriging_Model2 = SGEKriging_fit_2(inputpar,hyperpar);
t2 = clock;
Time2(ii,k) = etime(t2,t1)

[Mean Variance] = GEKriging_predictor(xtest,GEKriging_Model2);
MSE2(ii,k) = mean((Mean-y1).^2)/var(y1)


t1 = clock;
  GEKriging_Model3 = SGEKriging_fit_3(inputpar,hyperpar);
t2 = clock;

Time3(ii,k) = etime(t2,t1)
[Mean Variance] = GEKriging_predictor(xtest,GEKriging_Model3);
MSE3(ii,k) = mean((Mean-y1).^2)/var(y1)

end
end
