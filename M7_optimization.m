clc;  clear;

format long; syms x1 x2;

g = @(x)(15.*x(:,2)-5.1/4/pi^2.*(15.*x(:,1)-5).^2+5/pi.*(15.*x(:,1)-5)-6).^2+10.*((1-1/8/pi).*cos(15.*x(:,1)-5)+1)+5.*x(:,1);

const{1} = @(x)0.2-x(:,1).*x(:,2);

n = 2; num_const = 1; num_obj = 1;

Pd{1} = @(x)2.*(15.*x(:,2)-5.1/4/pi^2.*(15.*x(:,1)-5).^2+5/pi.*(15.*x(:,1)-5)-6).*(-10.2/4/pi^2.*(15.*x(:,1)-5)*15+75/pi)-10.*(1-1/8/pi).*sin(55.*x(:,1)-5).*55+5;
Pd{2} = @(x)2.*(15.*x(:,2)-5.1/4/pi^2.*(15.*x(:,1)-5).^2+5/pi.*(15.*x(:,1)-5)-6).*15;

const_partial{1,1} = @(x)-x(:,2);
const_partial{1,2} = @(x)-x(:,1);

%% Initial sampling

lb = [0 0];  ub = [1 1]; N = 4;  

for kk = 1 : 1

 mu = zeros(1,n);  sigma = ones(1,n);
 x = normcdf(lhsnorm(mu,diag(sigma.^2),N));

%pp = sobolset(n,'Skip',5); x = net(pp,N);  

y_obj = g(x);    
 
for i = 1:N
   Par = [];
  for j = 1:n
    Par_y(i) = Pd{j}(x(i,:));
    Par = [Par Par_y(i)];
  end
  grad_y(i,:) = Par;
end
 
 y_const{1} = const{1}(x); 

 grad_const{1} = [const_partial{1,1}(x) const_partial{1,2}(x) ];

%% Optimization

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
inputpar.y_const    = y_const;    % constraint sample
inputpar.grad_const = grad_const; % constraint gradient sample
inputpar.num_const  = 1;

subopt.cost          = 20;       % computational budget
subopt.objective     = g;        % objective function
subopt.obj_partial   = Pd;       % partial derivative of objective function
subopt.const         = const;    % contraint function
subopt.const_partial = const_partial;  % partial derivative of contraint function
subopt.method        = 'CMAES';   % EI function optimization method

subopt.acqfun        = 'EI';      % Acqiusition function

t1 = clock;
  [x_design1 objevtive1(kk,:) GEKriging_obj1 GEKriging_const] = GEKriging_optimization(inputpar,hyperpar,subopt);  % Bayesian optimization
t2 = clock;

end


 
