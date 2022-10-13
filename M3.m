clc;  clear;

syms x1 x2 x3 x4 x5 x6 x7 x8 x9 x10

c=[-6.089 -17.164 -34.054 -5.914 -24.721 -14.986 -24.1 -10.708 -26.662 -22.179];

Sum=@(x)x(:,1).^2+x(:,2).^2+x(:,3).^2+x(:,4).^2+x(:,5).^2+x(:,6).^2+x(:,7).^2+x(:,8).^2+x(:,9).^2+x(:,10).^2;
g=@(x)x(:,1).*(c(1)+log(x(:,1).^2./(Sum(x))))+x(:,2).*(c(2)+log(x(:,2).^2./(Sum(x))))+x(:,3).*(c(3)+log(x(:,3).^2./(Sum(x))))+x(:,4).*(c(4)+log(x(:,4).^2./(Sum(x))))+x(:,5).*(c(5)+log(x(:,5).^2./(Sum(x))))+x(:,6).*(c(6)+log(x(:,6).^2./(Sum(x))))+x(:,7).*(c(7)+log(x(:,7).^2./(Sum(x))))+x(:,8).*(c(8)+log(x(:,8).^2./(Sum(x))))+x(:,9).*(c(9)+log(x(:,9).^2./(Sum(x))))+x(:,10).*(c(10)+log(x(:,10).^2./(Sum(x))));
  
Sum1=x1.^2+x2.^2+x3.^2+x4.^2+x5.^2+x6.^2+x7.^2+x8.^2+x9.^2+x10.^2;
G=x1.*(c(1)+log(x1.^2./(Sum1)))+x2.*(c(2)+log(x2.^2./(Sum1)))+x3.*(c(3)+log(x3.^2./(Sum1)))+x4.*(c(4)+log(x4.^2./(Sum1)))+x5.*(c(5)+log(x5.^2./(Sum1)))+x6.*(c(6)+log(x6.^2./(Sum1)))+x7.*(c(7)+log(x7.^2./(Sum1)))+x8.*(c(8)+log(x8.^2./(Sum1)))+x9.*(c(9)+log(x9.^2./(Sum1)))+x10.*(c(10)+log(x10.^2./(Sum1)));

n=10;

grad_f=[diff(G,x1),diff(G,x2),diff(G,x3),diff(G,x4),diff(G,x5),diff(G,x6),diff(G,x7),diff(G,x8) ,diff(G,x9),diff(G,x10) ];

for i=1:n
   Pd{i}=matlabFunction(grad_f(i));
end

%% Sampling

Lb = -3.*ones(n,1)';  Ub = 3.*ones(n,1)'; N=50; N1=1000; 

Samplesize = 10 : 10: 50;

for k = 1: 5
 
x = [];

N = Samplesize(k); N1=3000; n=10;

pp = sobolset(n,'Skip',3); u=net(pp,N);  
pp1 = sobolset(n,'Skip',1000,'Leap',N1); u1=net(pp1,N1);  

for i=1:n
  x(:,i)=u(:,i)*(Ub(i)-Lb(i))+Lb(i);
  xtest(:,i)=u1(:,i)*(Ub(i)-Lb(i))+Lb(i);
end

 y=g(x); y1=g(xtest);
 
for i=1:N
   Par=[];
  for j=1:n
    Par_output(i)=Pd{j}(x(i,1),x(i,2),x(i,3),x(i,4),x(i,5),x(i,6),x(i,7),x(i,8),x(i,9),x(i,10));
    Par=[Par Par_output(i)];
  end
  grad_y(i,:)=Par;
end

%% GE-Kriging

hyperpar.theta = 0.1.*ones(1,n); 
hyperpar.lb = 5*10^-4.*ones(1,n);
hyperpar.ub = 5*ones(1,n);
hyperpar.corr_fun = 'corrbiquadspline';
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

Time(k) = etime(t2,t1)
[Mean Variance] = GEKriging_predictor(xtest,GEKriging_Model);
MSE(k)  = mean((Mean-y1).^2)/var(y1)

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

Time1(k) = etime(t2,t1)
[Mean Variance] = Kriging_predictor(xtest,Kriging_Model);
MSE1(k)  = mean((Mean-y1).^2)/var(y1)

%% Sliced GE-Kriging 

hyperpar.beta = [1 0.5 10^-2];
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

Time2(k) = etime(t2,t1)
[Mean Variance] = GEKriging_predictor(xtest,GEKriging_Model2);
MSE2(k) = mean((Mean-y1).^2)/var(y1)

t1 = clock;
  GEKriging_Model3 = SGEKriging_fit_3(inputpar,hyperpar);
t2 = clock;

Time3(k) = etime(t2,t1)
[Mean Variance] = GEKriging_predictor(xtest,GEKriging_Model3);
MSE3(k) = mean((Mean-y1).^2)/var(y1)

end

