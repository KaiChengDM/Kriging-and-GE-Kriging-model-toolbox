clc;  clear;

format long;

n = 10;

%%
%  if n > 2
%   g = @(x)sum((1-x(:,1:n-1)).^2')' + sum(100.*(x(:,2:n)-x(:,1:n-1).^2).^2')';
%  else
%   g = @(x)((1-x(:,1:n-1)).^2')' + (100.*(x(:,2:n)-x(:,1:n-1).^2).^2')';
%  end
% 
%  for i = 1:n-1
%   Pd{i} = @(x) -2.*(1-x(:,i)) - 400.*(x(:,i+1)-x(:,i).^2).*x(:,i);
%  end
% 
%  Pd{n} = @(x) 200.*(x(:,n)-x(:,n-1).^2);
%%
 g = @(x)0.5*sum(x.^4'-16.*x.^2'+5.*x')';

 for i = 1:n
  Pd{i} = @(x) 0.5*(4.*x(:,i).^3-32.*x(:,i)+5);
 end
%% Sampling

lb = -5.*ones(1,n);  ub = 5.*ones(1,n); N = 30; N1 = 3000; 

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

hyperpar.beta = [0.5 0.5 0.01];
hyperpar.lb   = [5.*10^-4  0.2 5.*10^-4]; 
hyperpar.ub   =  [2.5 1 2.5 ]; 

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
inputpar.snum       = 10;

subopt.cost         = 50;
subopt.objective    = g;
subopt.obj_partial  = Pd;
subopt.const        = [];
subopt.const_partial= [];
subopt.method       = 'CMAES';
subopt.acqfun       = 'EI';


t1 = clock;
  [x_design objevtive GEKriging_obj GEKriging_const] = GEKriging_optimization1(inputpar,hyperpar,subopt);  %% Sliced GE-Kriging for Bayesian optimization
t2 = clock;

