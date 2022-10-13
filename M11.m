clc;  clear;

format long;

n =10;

if n > 2
  g = @(x)sum((1-x(:,1:n-1)).^4')' + sum(((x(:,2:n)-x(:,1:n-1).^2).^2.*sqrt(1:n-1))')';
else
  g = @(x)((1-x(:,1:n-1)).^4')' + ((x(:,2:n)-x(:,1:n-1).^2).^2')';
end

for i = 1:n-1
  Pd{i} = @(x) -4.*(1-x(:,i)).^3 - 4*sqrt(i).*(x(:,i+1)-x(:,i).^2).*x(:,i);
end
Pd{n} = @(x) 2*sqrt(n-1).*(x(:,n)-x(:,n-1).^2);

%% 
%  g = @(x)0.5*sum(x.^4'-16.*x.^2'+5.*x')';
% 
%  for i = 1:n
%   Pd{i} = @(x) 0.5*(4.*x(:,i).^3-32.*x(:,i)+5);
%  end

%% Sampling

Samplesize = 20 : 20: 100;
sig = ones(1,n); mu = zeros(1,n);

for ii = 1 : 1
    u{ii} = normcdf(lhsnorm(mu,diag(sig.^2),10^4));
for k = 1 : 5
 
 x = []; y = []; grad_y = [];

 lb = -5.*ones(1,n);  ub = 5.*ones(1,n); N = Samplesize(k); N1 = 3000; 

%  pp = sobolset(n,'Skip',3); u=net(pp,N);  
%  pp1 = sobolset(n,'Skip',100,'Leap',N1); u1=net(pp1,N1); 

 for i=1:n
    x(:,i) = u{ii}(1:N,i)*(ub(i)-lb(i))+lb(i);
    xtest(:,i) = u{ii}(N+1:N+N1,i)*(ub(i)-lb(i))+lb(i);
 end

y = g(x); y1 = g(xtest);
 
for i = 1:N
   Par =[];
  for j = 1:n
    Par_output(i) = Pd{j}(x(i,:));
    Par = [Par Par_output(i)];
  end
  grad_y(i,:) = Par;
end

for i = 1:N1
   Par =[];
  for j = 1:n
    Par_output1(i) = Pd{j}(xtest(i,:));
    Par = [Par Par_output1(i)];
  end
  grad_y1(i,:) = Par;
end

%% GE-Kriging
% 
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

Time1(ii,k) = etime(t2,t1)
[Mean Variance] = GEKriging_predictor(xtest,GEKriging_Model);
MSE1(ii,k)  = mean((Mean-y1).^2)/var(y1)


%% Kriging

hyperpar.theta = 0.1.*ones(1,n); 
hyperpar.lb = 10^-3.*ones(1,n);
hyperpar.ub = 5*ones(1,n);
hyperpar.corr_fun = 'corrbiquadspline';
hyperpar.opt_algorithm = 'Hooke-Jeeves';
hyperpar.multistarts = 10;

inputpar.x = x;
inputpar.y = y;

t1=clock;
  Kriging_Model = Kriging_fit(inputpar,hyperpar);
t2=clock;

Time1(ii,k) = etime(t2,t1)
[Mean Variance] = Kriging_predictor(xtest,Kriging_Model);
MSE1(ii,k)  = mean((Mean-y1).^2)/var(y1)

%% Sliced GE-Kriging

hyperpar.beta = [0.5 0.5 10^-2];
hyperpar.lb   = [5*10^-4 0.2 5*10^-4]; 
hyperpar.ub   = [2.5 1 2.5];

hyperpar.corr_fun = 'corrbiquadspline';
%hyperpar.corr_fun = 'corrgaussian';
% hyperpar.opt_algorithm = 'Fmincon';
hyperpar.opt_algorithm = 'Hooke-Jeeves';
hyperpar.multistarts = 5;
inputpar.threshold = 1;
inputpar.x = x;
inputpar.y = y;
inputpar.grad = grad_y;
inputpar.snum = 10;

t1 = clock;
  GEKriging_Model2 = SGEKriging_fit_2(inputpar,hyperpar);
t2 = clock;
Time2(ii,k) = etime(t2,t1)

[Mean Variance] = GEKriging_predictor(xtest,GEKriging_Model2);
MSE2(ii,k) = mean((Mean-y1).^2)/var(y1)


t1 = clock;
  GEKriging_Model3 = SGEKriging_fit_3(inputpar,hyperpar);
t2 = clock;

Time3(ii,k) = etime(t2,t1)
[Mean Variance] = GEKriging_predictor(xtest,GEKriging_Model3);
MSE3(ii,k) = mean((Mean-y1).^2)/var(y1)

end
end


for i = 1 : 5
    Time_t(:,(i-1)*4+(1:4)) = [Time(:,i) Time1(:,i) Time2(:,i) Time3(:,i)];
    RSME_t(:,(i-1)*4+(1:4)) = [MSE(:,i) MSE1(:,i) MSE2(:,i) MSE3(:,i)];
end

subplot(1,2,1)
boxplot(Time_t); hold on
plot(1:4:20,median(Time_t(:,1:4:20)),'-o','LineWidth',2); hold on
plot(2:4:20,median(Time_t(:,2:4:20)),'-o','LineWidth',2); hold on
plot(3:4:20,median(Time_t(:,3:4:20)),'-o','LineWidth',2); hold on
plot(4:4:20,median(Time_t(:,4:4:20)),'-o','LineWidth',2); hold on

legend('GE-Kriging','Kriging','2-order SGE-Kriging','3-order SGE-Kriging')
xlabel('Model evaluations'); ylabel('Training time (s)')


subplot(1,2,2)
boxplot(RSME_t); hold on
plot(1:4:20,median(RSME_t(:,1:4:20)),'-o','LineWidth',2); hold on
plot(2:4:20,median(RSME_t(:,2:4:20)),'-o','LineWidth',2); hold on
plot(3:4:20,median(RSME_t(:,3:4:20)),'-o','LineWidth',2); hold on
plot(4:4:20,median(RSME_t(:,4:4:20)),'-o','LineWidth',2); hold on

legend('GE-Kriging','Kriging','2-order SGE-Kriging','3-order SGE-Kriging')
xlabel('Model evaluations'); ylabel('RMSE')


