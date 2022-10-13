clc;  clear;

format long;

syms x1 x2 x3 x4 x5 x6 x7 x8

g=@(x)2.*pi.*x(:,3).*(x(:,4)-x(:,6))./(log(x(:,2)./x(:,1)).*(1+2.*x(:,7).*x(:,3)./(log(x(:,2)./x(:,1)).*x(:,1).^2.*x(:,8))+x(:,3)./x(:,5)));

G=2.*pi.*x3.*(x4-x6)./(log(x2./x1).*(1+2.*x7.*x3./(log(x2./x1).*x1.^2.*x8)+x3./x5));

grad_f=[diff(G,x1),diff(G,x2),diff(G,x3),diff(G,x4),diff(G,x5),diff(G,x6),diff(G,x7),diff(G,x8)];  % Partial derivative

Pd1=matlabFunction(grad_f(1));
Pd2=matlabFunction(grad_f(2));
Pd3=matlabFunction(grad_f(3));
Pd4=matlabFunction(grad_f(4));
Pd5=matlabFunction(grad_f(5));
Pd6=matlabFunction(grad_f(6));
Pd7=matlabFunction(grad_f(7));
Pd8=matlabFunction(grad_f(8));

Lb=[0.05  100   63070  990  63.1 700 1120  9855 ];  % input lower bound 
Ub=[0.15 50000 115600  1110 116  820 1680 12045 ];  % input upper bound

%% Sampling

Samplesize = 10 :10: 50;

for k = 1: 5
 
x = [];

N = Samplesize(k);  N1=5000;   % sample size

n=8;   % input dimension

pp = sobolset(n,'Skip',3); u=net(pp,N);   % Sobol seqence 
pp1 = sobolset(n,'Skip',100,'Leap',N1); u1=net(pp1,N1);  

for i=1:n
  x(:,i)=u(:,i)*(Ub(i)-Lb(i))+Lb(i);
  xtest(:,i)=u1(:,i)*(Ub(i)-Lb(i))+Lb(i);
end

y=g(x);   y1=g(xtest);   % Input and output of training samples set N

Grad_output=[];

for i=1:N

% Par_output_11(i)=Pd1(xtest(i,1),xtest(i,2),xtest(i,3),xtest(i,4),xtest(i,5),xtest(i,6),xtest(i,7),xtest(i,8));
Par_output_1(i)=Pd1(x(i,1),x(i,2),x(i,3),x(i,4),x(i,5),x(i,6),x(i,7),x(i,8));
Par_output_2(i)=Pd2(x(i,1),x(i,2),x(i,3),x(i,4),x(i,5),x(i,6),x(i,7),x(i,8));
Par_output_3(i)=Pd3(x(i,1),x(i,2),x(i,3),x(i,4),x(i,5),x(i,6),x(i,7),x(i,8));
Par_output_4(i)=Pd4(x(i,1),x(i,2),x(i,3),x(i,5),x(i,7),x(i,8));
Par_output_5(i)=Pd5(x(i,1),x(i,2),x(i,3),x(i,4),x(i,5),x(i,6),x(i,7),x(i,8));
Par_output_6(i)=Pd6(x(i,1),x(i,2),x(i,3),x(i,5),x(i,7),x(i,8));
Par_output_7(i)=Pd7(x(i,1),x(i,2),x(i,3),x(i,4),x(i,5),x(i,6),x(i,7),x(i,8));
Par_output_8(i)=Pd8(x(i,1),x(i,2),x(i,3),x(i,4),x(i,5),x(i,6),x(i,7),x(i,8));

grad_y(i,:)=[Par_output_1(i) Par_output_2(i)  Par_output_3(i)  Par_output_4(i) Par_output_5(i) Par_output_6(i) Par_output_7(i) Par_output_8(i)];

end

%% GE-Kriging

hyperpar.theta = 0.1.*ones(1,n);   % Hyper-parameter 
hyperpar.lb = 5*10^-4.*ones(1,n);  % Lower bound of Hyper-parameter 
hyperpar.ub = 5*ones(1,n);         % Upper bound of Hyper-parameter 
hyperpar.corr_fun = 'corrbiquadspline';  % Correlation function
% hyperpar.corr_fun = 'corrgaussian';
hyperpar.opt_algorithm = 'Hooke-Jeeves'; % Hyper-parameter tuning method
hyperpar.multistarts = 10;               % Multi-starts
 
inputpar.lb = Lb;              % Input lower bound
inputpar.ub = Ub;              % Input upper bound
inputpar.x = x;                % input samples
inputpar.y = y;                % output samples
inputpar.grad = grad_y;        % gradient samples

t1=clock;
 GEKriging_Model = GEKriging_fit(inputpar,hyperpar);  % training GE-Kriging 
t2=clock;

Time(k) = etime(t2,t1)
[Mean Variance] = GEKriging_predictor(xtest,GEKriging_Model);  % making prediction
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
[Mean Variance] = Kriging_predictor(xtest,Kriging_Model);
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

