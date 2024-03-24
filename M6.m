clc;  clear;

format long;
syms x1 x2;

g=@(x)(4-2.1.*x(:,1).^2+x(:,1).^4./3).*x(:,1).^2+x(:,1).*x(:,2)+(-4+4.*x(:,2).^2).*x(:,2).^2;
G=(4-2.1.*x1.^2+x1.^4./3).*x1.^2+x1.*x2+(-4+4.*x2.^2).*x2.^2;

grad_f=[diff(G,x1),diff(G,x2)];

for i=1:2
  Pd{i}=matlabFunction(grad_f(i));
end

%% Sampling

Lb=[-2 -1];  Ub=[2 1]; N = 20; N1=1000;  n=2;

pp = sobolset(n,'Skip',3); u = net(pp,N);  
pp1 = sobolset(n,'Skip',100,'Leap',N1); u1 = net(pp1,N1);  

for i=1:n
  x(:,i) = u(:,i)*(Ub(i)-Lb(i))+Lb(i);
  xtest(:,i) = u1(:,i)*(Ub(i)-Lb(i))+Lb(i);
end

y=g(x); y1=g(xtest);
 
for i=1:N
   Par=[];
  for j=1:n
    Par_output(i)=Pd{j}(x(i,1),x(i,2));
    Par=[Par Par_output(i)];
  end
  grad_y(i,:)=Par;
end

%% Training GE-Kriging

hyperpar.theta = 0.1.*ones(1,n); 
hyperpar.lb = 5*10^-4.*ones(1,n);
hyperpar.ub = 100*ones(1,n);
hyperpar.corr_fun = 'corrbiquadspline';
%hyperpar.corr_fun = 'corrgaussian';
hyperpar.opt_algorithm = 'Hooke-Jeeves';
hyperpar.multistarts = 10;

inputpar.lb = Lb;
inputpar.ub = Ub;
inputpar.x = x;
inputpar.y = y;
inputpar.grad = grad_y;

t1=clock;
 GEKriging_Model = GEKriging_fit(inputpar,hyperpar);
t2=clock;
etime(t2,t1)

Time = etime(t2,t1)
[Mean, Var] = GEKriging_predictor(xtest,GEKriging_Model);
MSE = mean((Mean-y1).^2)/var(y1)

%% Training Sliced GE-Kriging

hyperpar.beta = [0.5 0.5 10^-2];
hyperpar.lb   = [5*10^-4 0.2 5*10^-4]; 
hyperpar.ub   = [2.5 1 2.5];

hyperpar.corr_fun = 'corrbiquadspline';
% hyperpar.corr_fun = 'corrgaussian';
% hyperpar.corr_fun = 'corrspline';
hyperpar.opt_algorithm = 'Hooke-Jeeves';
hyperpar.multistarts = 10;
inputpar.threshold = 1;
inputpar.x = x;
inputpar.y = y;
inputpar.grad = grad_y;
inputpar.snum = 5;

t1 = clock;
  GEKriging_Model1 = SGEKriging_fit_2(inputpar,hyperpar);
t2 = clock;
etime(t2,t1)

Time = etime(t2,t1)
[Mean, Var] = GEKriging_predictor(xtest,GEKriging_Model1);
MSE1 = mean((Mean-y1).^2)/var(y1)

% t1 = clock;
%   GEKriging_Model2 = SGEKriging_fit_3(inputpar,hyperpar);
% t2 = clock;
% etime(t2,t1)
% 
% Time = etime(t2,t1)
% [Mean Variance] = GEKriging_predictor(xtest,GEKriging_Model2);
% MSE2 = mean((Mean-y1).^2)/var(y1)
