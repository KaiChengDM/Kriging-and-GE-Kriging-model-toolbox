clc;  clear;

format long;

g=@(x)sin(x(:,1))+7.*sin(x(:,2)).^2+0.1.*(x(:,3)).^4.*sin(2*pi*x(:,1));

Pd{1} = @(x) cos(x(:,1))+2*pi*0.1.*(x(:,3)).^4.*cos(2*pi*x(:,1));
Pd{2} = @(x) 14.*cos(x(:,2)).*sin(x(:,2));
Pd{3} = @(x) 0.4.*(x(:,3)).^3.*sin(2*pi*x(:,1));

n =3;

%% GEK optimization

for kk = 1:1
 
 sig = ones(1,n); mu = zeros(1,n);
 lb = -2*pi.*ones(1,n);  ub = 2*pi.*ones(1,n); N = 3;  N1 = 1000;
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

subopt.cost          = 100;
subopt.objective     = g;
subopt.obj_partial   = Pd;
subopt.const         = [];
subopt.const_partial = [];
subopt.method        = 'CMAES'; 

subopt.acqfun        = 'EI';
t1 = clock;
  [x_design1 objevtive1(kk,:) GEKriging_obj GEKriging_const] = GEKriging_optimization(inputpar,hyperpar,subopt);
t2 = clock;
end

 