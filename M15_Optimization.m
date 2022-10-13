clc; clear all; 

g = @(x)-(1+cos(6.*sqrt(x(:,1).^2+x(:,2).^2)))./(0.5.*(x(:,1).^2+x(:,2).^2)+2) % Drop-wave FUNCTION

Pd{1} = @(x)(x(:,1)*(cos(12*(x(:,1)^2 + x(:,2)^2)^(1/2)) + 1))/(x(:,1)^2/2 + x(:,2)^2/2 + 2)^2 + (12*x(:,1)*sin(12*(x(:,1)^2 + x(:,2)^2)^(1/2)))/((x(:,1)^2 + x(:,2)^2)^(1/2)*(x(:,1)^2/2 + x(:,2)^2/2 + 2));

Pd{2} = @(x)(x(:,2)*(cos(12*(x(:,1)^2 + x(:,2)^2)^(1/2)) + 1))/(x(:,1)^2/2 + x(:,2)^2/2 + 2)^2 + (12*x(:,2)*sin(12*(x(:,1)^2 + x(:,2)^2)^(1/2)))/((x(:,1)^2 + x(:,2)^2)^(1/2)*(x(:,1)^2/2 + x(:,2)^2/2 + 2));

 n = 2;

%% 
 for kk = 1:20

 sig = ones(1,n); mu = zeros(1,n);
 lb = -3.*ones(1,n);  ub = 3.*ones(1,n); N = 20;  N1 = 1000;
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

subopt.cost          = 20;
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

