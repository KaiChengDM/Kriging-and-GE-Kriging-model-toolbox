clc;  clear;

format long;

n = 10;

g = @(x)(sum((x.*(1:n))')+sum((x.^3.*(1:n))')+log(sum(((x.^2+x.^4).*(1:n))')))';

for i = 1 : n
  Pd{i} = @(x)i+3.*i*x(:,i).^2+(i*(2.*x(:,i)+4.*x(:,i).^3))./(sum(((x.^2+x.^4).*(1:n))')');
end

%% Sampling

lb = -5.*ones(1,n);  ub = 5.*ones(1,n); N = 20; N1 = 3000;

% pp = sobolset(n,'Skip',5); u=net(pp,N);  
% pp1 = sobolset(n,'Skip',1001,'Leap',N1); u1=net(pp1,N1);  

sig = ones(1,n); mu = zeros(1,n);
u = normcdf(lhsnorm(mu,diag(sig.^2),N));
u1 = normcdf(lhsnorm(mu,diag(sig.^2),N1));

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

% hyperpar.beta = [1 0.5 10^-2];
% hyperpar.lb   = [5*10^-4 0.2 5*10^-4]; 
% hyperpar.ub   =  [2.5 1 2.5];

hyperpar.corr_fun       = 'corrbiquadspline';
% hyperpar.corr_fun     = 'corrgaussian';
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

subopt.cost          = 100;
subopt.objective     = g;
subopt.obj_partial   = Pd;
subopt.const         = [];
subopt.const_partial = [];
subopt.method        = 'GA'; 
 
subopt.acqfun        = 'EI';
t1 = clock;
  [x_design objevtive GEKriging_obj GEKriging_const index] = GEKriging_optimization(inputpar,hyperpar,subopt);
t2 = clock;

