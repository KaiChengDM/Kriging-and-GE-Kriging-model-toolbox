
clc;  clear;

g=@(x)((log(x(:,1).^2)).^2+(log(x(:,1))).^2)+((log(x(:,2).^2)).^2+(log(x(:,2))).^2)+((log(x(:,3).^2)).^2+(log(x(:,3))).^2)+((log(x(:,4).^2)).^2+(log(x(:,4))).^2)+((log(x(:,5).^2)).^2+(log(x(:,5))).^2)-(x(:,1).*x(:,2).*x(:,3).*x(:,4).*x(:,5)).^0.2;

syms x1 x2 x3 x4 x5 

G=((log(x1.^2)).^2+(log(x1)).^2)+((log(x2.^2)).^2+(log(x2)).^2)+((log(x3.^2)).^2+(log(x3)).^2)+((log(x4.^2)).^2+(log(x4)).^2)+((log(x5.^2)).^2+(log(x5)).^2)-(x1.*x2.*x3.*x4.*x5).^0.2;

grad_f=[diff(G,x1),diff(G,x2),diff(G,x3),diff(G,x4),diff(G,x5)];

Pd1=matlabFunction(grad_f(1));
Pd2=matlabFunction(grad_f(2));
Pd3=matlabFunction(grad_f(3));
Pd4=matlabFunction(grad_f(4));
Pd5=matlabFunction(grad_f(5));

lb = ones(1,5);
ub = 10.*ones(1,5);

%% Sampling

Samplesize = 10 : 10: 50;

for k = 1: 5
 
x = [];

N = Samplesize(k); N1=3000; n=5;

pp = sobolset(n,'Skip',3); u=net(pp,N);  
pp1 = sobolset(n,'Skip',1000,'Leap',N1); u1=net(pp1,N1);  

for i=1:n
  x(:,i)=u(:,i)*(ub(i)-lb(i))+lb(i);
  xtest(:,i)=u1(:,i)*(ub(i)-lb(i))+lb(i);
end

y=g(x); y1=g(xtest);
 
for i=1:N

Par_output_1(i)=Pd1(x(i,1),x(i,2),x(i,3),x(i,4),x(i,5));
Par_output_2(i)=Pd2(x(i,1),x(i,2),x(i,3),x(i,4),x(i,5));
Par_output_3(i)=Pd3(x(i,1),x(i,2),x(i,3),x(i,4),x(i,5));
Par_output_4(i)=Pd4(x(i,1),x(i,2),x(i,3),x(i,4),x(i,5));
Par_output_5(i)=Pd5(x(i,1),x(i,2),x(i,3),x(i,4),x(i,5));

grad_y(i,:)=[Par_output_1(i) Par_output_2(i)  Par_output_3(i)  Par_output_4(i) Par_output_5(i)];

end

%% GE-Kriging

hyperpar.theta = 0.1.*ones(1,n); 
hyperpar.lb = 5*10^-4.*ones(1,n);
hyperpar.ub = 5*ones(1,n);
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

Time(k) = etime(t2,t1)
[Mean Variance] = GEKriging_predictor(xtest,GEKriging_Model);
MSE(k)  = mean((Mean-y1).^2)/var(y1)

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

Time1(k) = etime(t2,t1)
[Mean, Variance] = Kriging_predictor(xtest,Kriging_Model);
MSE1(k)  = mean((Mean-y1).^2)/var(y1)

%% Sliced GE-Kriging 

hyperpar.beta = [1 0.5 10^-2];
hyperpar.lb   = [5*10^-4 0.2 5*10^-4]; 
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


