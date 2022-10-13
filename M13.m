clc; clear all; 

g = @(x)(sum(((1:5).*cos((2:6).*x(:,1)+(1:5)))').*sum(((1:5).*cos((2:6).*x(:,2)+(1:5)))'))';

Pd{1} = @(x)(sum((-(1:5).*(2:6).*sin((2:6).*x(:,1)+(1:5)))').*sum(((1:5).*cos((2:6).*x(:,2)+(1:5)))'))';
Pd{2} = @(x)(sum(((1:5).*cos((2:6).*x(:,1)+(1:5)))').*sum((-(1:5).*(2:6).*sin((2:6).*x(:,2)+(1:5)))'))';

 n = 2;

%  lb = -10.*ones(1,n);  ub =10.*ones(1,n); 
%  nn=400;
%  xx = lb(1):(ub(1)-lb(1))/(nn-1):ub(1);
%  yy = lb(2):(ub(2)-lb(2))/(nn-1):ub(2);
%  [X,Y] = meshgrid(xx,yy);
%  figure
%  xnod  = cat(2,reshape(X',nn^2,1),reshape(Y',nn^2,1));
%  ZZ= g(xnod); ZZ=reshape(ZZ,nn,nn);  mesh(X,Y,ZZ');

%% 
Samplesize = 20 : 20: 100;
sig = ones(1,n); mu = zeros(1,n);

for ii = 1 : 10
    u{ii} = normcdf(lhsnorm(mu,diag(sig.^2),10^4));
for k = 1 : 5
 
 x = []; y = []; grad_y = [];

 lb = -3.*ones(1,n);  ub = 3.*ones(1,n); N = Samplesize(k); N1 = 3000; 

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
hyperpar.ub = 5*ones(1,n);
hyperpar.corr_fun = 'corrbiquadspline';
hyperpar.opt_algorithm = 'Hooke-Jeeves';
hyperpar.multistarts = 10;

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

hyperpar.theta = 0.1.*ones(1,n); 
hyperpar.lb = 5*10^-4.*ones(1,n);
hyperpar.ub = 5*ones(1,n);
hyperpar.corr_fun = 'corrbiquadspline';
hyperpar.opt_algorithm = 'Hooke-Jeeves';
hyperpar.multistarts = 10;

inputpar.x = x;
inputpar.y = y;

t1=clock;
  Kriging_Model = Kriging_fit(inputpar,hyperpar);
t2=clock;

Time1(ii,k) = etime(t2,t1)
[Mean Variance] = Kriging_predictor(xtest,Kriging_Model);
MSE1(ii,k)  = mean((Mean-y1).^2)/var(y1)

%% Sliced GE-Kriging

hyperpar.beta = [0.5 0.5 10^-2];
hyperpar.lb   = [5*10^-4 0.2 5*10^-4]; 
hyperpar.ub   = [2.5 1 2.5];

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
