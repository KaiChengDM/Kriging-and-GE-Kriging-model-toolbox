clc;  clear;

format long;

g=@(x)sin(x(:,1))+7.*sin(x(:,2)).^2+0.1.*(x(:,3)).^4.*sin(x(:,1));

Pd{1} = @(x) cos(x(:,1))+0.1.*(x(:,3)).^4.*cos(x(:,1));
Pd{2} = @(x) 14.*cos(x(:,2)).*sin(x(:,2));
Pd{3} = @(x) 0.4.*(x(:,3)).^3.*sin(x(:,1));

n =3;

%% Sampling

Samplesize = 50 : 20 : 150;

sig = ones(1,n); mu = zeros(1,n);

for ii = 1 : 1

 u = normcdf(lhsnorm(mu,diag(sig.^2),10^4));

for k = 1 : 5
 
x = []; y = []; grad_y = [];

lb=-pi.*ones(1,n);  ub=pi.*ones(1,n); N = Samplesize(k); N1=1000; 

pp = sobolset(n,'Skip',3); u=net(pp,N);  
pp1 = sobolset(n,'Skip',1000,'Leap',N1); u1=net(pp1,N1);  

for i=1:n
  x(:,i)=u(:,i)*(ub(i)-lb(i))+lb(i);
  xtest(:,i)=u1(:,i)*(ub(i)-lb(i))+lb(i);
end

y=g(x); y1=g(xtest);
 
for i=1:N
   Par=[];
  for j=1:n
    Par_output(i)=Pd{j}(x(i,:));
    Par=[Par Par_output(i)];
  end
  grad_y(i,:)=Par;
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
hyperpar.lb = 10^-3.*ones(1,n);
hyperpar.ub = 5*ones(1,n);
hyperpar.corr_fun = 'corrbiquadspline';
%hyperpar.corr_fun = 'corrspline';
%hyperpar.corr_fun = 'corrgaussian';
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
% 
hyperpar.theta = 0.1.*ones(1,n); 
hyperpar.lb = 10^-3.*ones(1,n);
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
hyperpar.lb   = [10^-3 0.2 10^-3]; 
hyperpar.ub   =  [2.5 1 2.5];

hyperpar.corr_fun = 'corrbiquadspline';
%hyperpar.corr_fun = 'corrgaussian';
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