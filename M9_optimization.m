clc;  clear;

format long;

n = 10;

g = @(x)((x(:,1)-1).^2+sum(((2*x(:,2:n).^2-x(:,1:n-1)).^2.*(2:n))')');
 
Pd{1} = @(x) (2.*(x(:,1)-1) -4*(2*x(:,2).^2-x(:,1)));

for i = 2 : n
  Pd{i} = @(x)(2*i*(2*x(:,i).^2-x(:,i-1)).*(4*x(:,i)));
end

%% Sampling

lb = -10.*ones(1,n);  ub = 10.*ones(1,n); N = 10; N1 = 3000; 

pp = sobolset(n,'Skip',3); u=net(pp,N);  
pp1 = sobolset(n,'Skip',100,'Leap',N1); u1=net(pp1,N1);  

for i=1:n
  x(:,i) = u(:,i)*(ub(i)-lb(i))+lb(i);
  xtest(:,i)=u1(:,i)*(ub(i)-lb(i))+lb(i);
end

y_obj = g(x); y1 = g(xtest);
 
for i = 1:N
   Par = [];
  for j=1:n
    Par_output(i)=Pd{j}(x(i,:));
    Par=[Par Par_output(i)];
  end
  grad_y(i,:)=Par;
end

%%  Adaptive Kriging for efficient global optimization

hyperpar.theta = 0.1.*ones(1,n); 
hyperpar.lb = 5*10^-4.*ones(1,n);
hyperpar.ub = 5*ones(1,n);
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

subopt.cost          = 30;
subopt.objective     = g;
subopt.obj_partial   = Pd;
subopt.const         = [];
subopt.const_partial = [];
subopt.method        = 'CMAES'; 

subopt.acqfun        = 'EI';
t1 = clock;
  [x_design1 objevtive1 GEKriging_obj GEKriging_const] = GEKriging_optimization(inputpar,hyperpar,subopt);
t2 = clock;
