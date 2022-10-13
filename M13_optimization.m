clc; clear all; 

g = @(x)(sum(((1:5).*cos((2:6).*x(:,1)+(1:5)))').*sum(((1:5).*cos((2:6).*x(:,2)+(1:5)))'))'; % SHUBERT FUNCTION

Pd{1} = @(x)(sum((-(1:5).*(2:6).*sin((2:6).*x(:,1)+(1:5)))').*sum(((1:5).*cos((2:6).*x(:,2)+(1:5)))'))';
Pd{2} = @(x)(sum(((1:5).*cos((2:6).*x(:,1)+(1:5)))').*sum((-(1:5).*(2:6).*sin((2:6).*x(:,2)+(1:5)))'))';

n = 2;

%% 
 for kk = 1:20
 
 sig = ones(1,n); mu = zeros(1,n);
 lb = -5.12.*ones(1,n);  ub = 5.12.*ones(1,n); N = 20;  N1 = 1000;
% pp = sobolset(n,'Skip',5); u=net(pp,N);  
 u = normcdf(lhsnorm(mu,diag(sig.^2),N));
 u1 = normcdf(lhsnorm(mu,diag(sig.^2),N1));

 for i = 1:n
    x(:,i) = u(1:N,i)*(ub(i)-lb(i))+lb(i);
    xtest(:,i) = u1(1:N1,i)*(ub(i)-lb(i))+lb(i);
 end

 y_obj = g(x);  y1 = g(xtest); 
 
 for i = 1:N
   Par =[];
  for j = 1:n
    Par_output(i) = Pd{j}(x(i,:));
    Par = [Par Par_output(i)];
  end
  grad_y(i,:) = Par;
 end
 
%% GEK Optimization

hyperpar.theta = 0.1.*ones(1,n); 
hyperpar.lb = 5*10^-4.*ones(1,n);
hyperpar.ub = 50*ones(1,n);
hyperpar.corr_fun       = 'corrbiquadspline';
hyperpar.opt_algorithm  = 'Hooke-Jeeves';
hyperpar.multistarts    = 10;   

inputpar.x          = x;
inputpar.lb         = lb;
inputpar.ub         = ub;
inputpar.y_obj      = y_obj;
inputpar.grad_y     = grad_y;
inputpar.y_const    = [];
inputpar.grad_const = [];
inputpar.threshold  = 1;
inputpar.num_const  = 0;

subopt.cost          = 10;
subopt.objective     = g;
subopt.obj_partial   = Pd;
subopt.const         = [];
subopt.const_partial = [];
subopt.method        = 'CMAES'; 
subopt.acqfun        = 'EI';
t1 = clock;
  [x_design1 objevtive1(kk,:) GEKriging_obj1 GEKriging_const1 EI1] = GEKriging_optimization(inputpar,hyperpar,subopt);
t2 = clock;


hyperpar.beta = [1 0.5 10^-2];
hyperpar.lb   = [5*10^-4 0.2 5*10^-4]; 
hyperpar.ub   =  [25 1 25];
inputpar.snum = 10;

t1 = clock;
  [x_design2 objevtive2(kk,:) GEKriging_obj2 GEKriging_const2 EI2] = GEKriging_optimization1(inputpar,hyperpar,subopt);
t2 = clock;


 end

