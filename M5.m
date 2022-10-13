
clc;  clear;

n=1;
 
g = @(x)exp(-x(:,1))+sin(5.*x(:,1))+cos(1.*x(:,1))+0.2.*x(:,1)+4;

Pd{1} = @(x)-exp(-x(:,1)) +5.* cos(5.*x(:,1))-1.*sin(1.*x(:,1))+0.2;

%% Sampling

 sig = ones(1,n); mu = zeros(1,n);
 lb = 0.*ones(1,n);  ub = 6.*ones(1,n); N = 10;  N1 = 1000;
 
 pp = sobolset(n,'Skip',10); u=net(pp,N);  

% u = normcdf(lhsnorm(mu,diag(sig.^2),N));
 u1 = normcdf(lhsnorm(mu,diag(sig.^2),N1));

 for i = 1:n
    x(:,i) = u(1:N,i)*(ub(i)-lb(i))+lb(i);
    xtest(:,i) = u1(1:N1,i)*(ub(i)-lb(i))+lb(i);
 end

y = g(x);  y1 = g(xtest); 
 
 for i = 1:N
   Par =[];
  for j = 1:n
    Par_output(i) = Pd{j}(x(i,:));
    Par = [Par Par_output(i)];
  end
  grad_y(i,:) = Par;
 end
%% Training GE-Kriging

hyperpar.theta = 0.1.*ones(1,n); 
hyperpar.lb = 5*10^-4.*ones(1,n);
hyperpar.ub = 10*ones(1,n);
hyperpar.corr_fun = 'corrbiquadspline';
%hyperpar.corr_fun = 'corrspline';
% hyperpar.corr_fun = 'corrgaussian';
hyperpar.opt_algorithm = 'Hooke-Jeeves';
% hyperpar.opt_algorithm = 'Fmincon';
hyperpar.multistarts = 10;

inputpar.x = x;
inputpar.y = y;
inputpar.grad = grad_y;
inputpar.lb = lb;
inputpar.ub = ub;

t1=clock;
 GEKriging_Model = GEKriging_fit(inputpar,hyperpar);
t2=clock;

%% Training Sliced GE-Kriging

% hyperpar.corr_fun = 'corrbiquadspline';
 hyperpar.corr_fun = 'corrspline';
%  hyperpar.corr_fun = 'corrgaussian';
hyperpar.opt_algorithm = 'Hooke-Jeeves';
hyperpar.multistarts = 10;
inputpar.threshold = 1;
inputpar.x = x;
inputpar.y = y;
inputpar.grad = grad_y;
inputpar.lb = lb;
inputpar.ub = ub;
inputpar.snum = 10;

t1 = clock;
  GEKriging_Model2 = GEKriging_fit21(inputpar,hyperpar);
t2 = clock;
t1 = clock;
  GEKriging_Model3 = GEKriging_fit31(inputpar,hyperpar);
t2 = clock;

Time2 = etime(t2,t1)
[Mean Variance] = GEKriging_predictor(xtest,GEKriging_Model2);
MSE2 = mean((Mean-y1).^2)/var(y1)


 %% Response 

 nn = 10000;
 xx = lb:(ub-lb)/(nn-1):ub;
 
 %plot(xx,g(xx'),'linewidth',1.5); hold on
 %plot(x,y,'ro','linewidth',1.5); hold on

[y1 v1] = GEKriging_predictor(xx',GEKriging_Model);
[y2 v2] = GEKriging_predictor(xx',GEKriging_Model2);
[y3 v3] = GEKriging_predictor(xx',GEKriging_Model3);

  subplot(1,3,1)
  plot(x,y,'ro','linewidth',2); hold on
  plot(xx,g(xx'),'b-','linewidth',2); hold on
  plot(xx,y1,'m-.','linewidth',2); hold on
  plot(xx,sqrt(v1),'Color',[0.5 0.1 0.8],'linewidth',2); hold on
  legend('Samples','True function','2-appendant SGE-Kriging mean','2-appendant SGE-Kriging variance')
  xlabel('x'); ylabel('y'); 

  subplot(1,3,2)
  plot(x,y,'ro','linewidth',2); hold on
  plot(xx,g(xx'),'b-','linewidth',2); hold on
  plot(xx,y2,'m-.','linewidth',2); hold on
  plot(xx,v2,'Color',[0.5 0.1 0.8],'linewidth',2); hold on
  legend('Samples','True function','2-appendant SGE-Kriging mean','2-appendant SGE-Kriging variance')
  xlabel('x'); ylabel('y'); 
  
  subplot(1,3,3)
  plot(x,y,'ro','linewidth',2); hold on
  plot(xx,g(xx'),'b-','linewidth',2); hold on
  plot(xx,y3,'m-.','linewidth',2); hold on
  plot(xx,v3,'Color',[0.5 0.1 0.8],'linewidth',2); hold on
  legend('Samples','True function','3-appendant SGE-Kriging mean','3-appendant SGE-Kriging variance')
  xlabel('x'); ylabel('y'); 


