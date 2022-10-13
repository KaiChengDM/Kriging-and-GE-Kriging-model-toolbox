clc;  clear;

format long;

n = 10;

g = @(x)((x(:,1)-1).^2+sum(((2*x(:,2:n).^2-x(:,1:n-1)).^2.*(2:n))')');
 
Pd{1} = @(x) (2.*(x(:,1)-1) -4*(2*x(:,2).^2-x(:,1)));

for i = 2 : n
  Pd{i} = @(x)(2*i*(2*x(:,i).^2-x(:,i-1)).*(4*x(:,i)));
end

%% Sampling

Samplesize = 20 : 20 : 100;

sig = ones(1,n); mu = zeros(1,n);

for ii = 1 : 10
  u{ii} = normcdf(lhsnorm(mu,diag(sig.^2),10^4));
for k = 1 : 5
 
 x = []; y = []; grad_y = [];

 lb = -10.*ones(1,n);  ub = 10.*ones(1,n); N = Samplesize(k); N1 = 1000; 

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

for i = 1:N1
   Par1 =[];
  for j = 1:n
     Par_output1(i) = Pd{j}(xtest(i,:));
     Par1 = [Par1 Par_output1(i)];
  end
  grad_y1(i,:) = Par1;
end

%% GE-Kriging 

hyperpar.theta = 0.1.*ones(1,n); 
hyperpar.lb = 5*10^-4.*ones(1,n);
hyperpar.ub = 5*ones(1,n);
%hyperpar.corr_fun = 'corrspline';
hyperpar.corr_fun = 'corrbiquadspline';
%hyperpar.corr_fun = 'corrgaussian';
hyperpar.opt_algorithm = 'Hooke-Jeeves';
hyperpar.multistarts =5;

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
% 
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

hyperpar.beta = [1 0.5 10^-2];
hyperpar.lb   = [5*10^-4 0.2 5*10^-4]; 
hyperpar.ub   =  [2.5 1 2.5];

hyperpar.corr_fun = 'corrbiquadspline';
%hyperpar.corr_fun = 'corrgaussian';
hyperpar.opt_algorithm = 'Hooke-Jeeves';
% hyperpar.opt_algorithm = 'CMAES';
hyperpar.multistarts = 5;
inputpar.threshold = 0.99;
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

ii = 1:5;
plot(ii,log10(Time(1,:)),'-o'); hold on
plot(ii,log10(Time1(1,:)),'-o'); hold on
plot(ii,log10(Time2(1,:)),'-o'); hold on
plot(ii,log10(Time3(1,:)),'-o'); hold on

figure
ii = 1:5;
plot(ii,log10(MSE(1,:)),'-o'); hold on
plot(ii,log10(MSE1(1,:)),'-o'); hold on
plot(ii,log10(MSE2(1,:)),'-o'); hold on
plot(ii,log10(MSE3(1,:)),'-o'); hold on

% (GEKriging_Model3.sensitivity./sum(GEKriging_Model3.sensitivity)).^1


