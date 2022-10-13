clc;  clear;

format long;
syms x1 x2;

g = @(x)-(x(:,2)+47).*sin(sqrt(abs(x(:,2)+x(:,1)./2+47)))-x(:,1).*sin(sqrt(abs(x(:,1)-(x(:,2)+47)))); % Performance function
G = -(x2+47).*sin(sqrt(abs(x2+x1./2+47)))-x1.*sin(sqrt(abs(x1-(x2+47))));

grad_f = [diff(G,x1),diff(G,x2)];
for i=1:2
   Pd{i}=matlabFunction(grad_f(i));  % Partial derivative function
end

%% Sampling

Lb=[-512 -512];  Ub=[512 512];   % Lower bound and upper bound
N = 200; N1 = 3000;  n=2;        % N: training sampling size; N1: test sampling size; n: dimension

sig = ones(1,n); mu = zeros(1,n);
u = normcdf(lhsnorm(mu,diag(sig.^2),N));  % Sampling with LHS
u1 = normcdf(lhsnorm(mu,diag(sig.^2),N1));

for i = 1:n
  x(:,i) = u(:,i)*(Ub(i)-Lb(i))+Lb(i);
  xtest(:,i)=u1(:,i)*(Ub(i)-Lb(i))+Lb(i);
end

y = g(x); y1 = g(xtest);  % model response
 
for i=1:N
   Par=[];
  for j=1:n
    Par_output(i) = Pd{j}(x(i,1),x(i,2));
    Par = [Par Par_output(i)];
  end
  grad_y(i,:)=Par;  % gradient 
end

%% Training GE-Kriging

hyperpar.theta = 0.1.*ones(1,n); 
hyperpar.lb = 5*10^-4.*ones(1,n);  % Lower bpund of hyper-parameters
hyperpar.ub = 10*ones(1,n);       % Upper bpund of hyper-parameters

hyperpar.corr_fun = 'corrbiquadspline';  % Correlation functions
%hyperpar.corr_fun = 'corrgaussian';
%hyperpar.corr_fun = 'corrspline';

hyperpar.opt_algorithm = 'Hooke-Jeeves'; % Hyper-parameter optimization algorithm
% hyperpar.opt_algorithm = 'Fmincon'; 
% hyperpar.opt_algorithm = 'GA'; 
% hyperpar.opt_algorithm = 'CMAES'; 

hyperpar.multistarts = 10;  % Multiple starts to avoid local optimal solutions

inputpar.lb = Lb;
inputpar.ub = Ub;
inputpar.x = x;
inputpar.y = y;
inputpar.grad = grad_y;

t1=clock;
    GEKriging_Model = GEKriging_fit(inputpar,hyperpar);  % Training GE-Kriging
t2=clock;
etime(t2,t1) 

Time = etime(t2,t1)
[Mean Variance] = GEKriging_predictor(xtest,GEKriging_Model);  % Making prediction
MSE = mean((Mean-y1).^2)/var(y1)

%% Training Sliced GE-Kriging

%hyperpar.corr_fun = 'corrbiquadspline';  % Correlation functions
%hyperpar.corr_fun = 'corrgaussian';
%hyperpar.corr_fun = 'corrspline';

hyperpar.opt_algorithm = 'Hooke-Jeeves'; % Hyper-parameter optimization algorithm
% hyperpar.opt_algorithm = 'Fmincon'; 
% hyperpar.opt_algorithm = 'GA'; 
% hyperpar.opt_algorithm = 'CMAES'; 

hyperpar.multistarts = 10;
inputpar.threshold = 1;
inputpar.x = x;
inputpar.y = y;
inputpar.grad = grad_y;
inputpar.snum = 10;  % Split the sampling sites into m slices

t1 = clock;
  GEKriging_Model1 = GEKriging_fit21(inputpar,hyperpar);  % Training 2 appendant sliced GE-Kriging
t2 = clock;
etime(t2,t1)

Time = etime(t2,t1)
[Mean Variance] = GEKriging_predictor(xtest,GEKriging_Model); % Making prediction
MSE1 = mean((Mean-y1).^2)/var(y1)

t1 = clock;
  GEKriging_Model2 = GEKriging_fit31(inputpar,hyperpar);  % Training 3 appendant sliced GE-Kriging
t2 = clock;
etime(t2,t1)

Time = etime(t2,t1)
[Mean Variance] = GEKriging_predictor(xtest,GEKriging_Model); % Making prediction
MSE2 = mean((Mean-y1).^2)/var(y1)

% %% Plot of response surface
% 
% nn = 200;
% xx = (Lb(1):(Ub(1)-Lb(1))/(nn-1):Ub(1));
% yy = (Lb(2):(Ub(2)-Lb(2))/(nn-1):Ub(2));
% [X,Y] = meshgrid(xx,yy);
% xnod  = cat(2,reshape(X',nn^2,1),reshape(Y',nn^2,1));
% 
% yy = g(xnod);
% [y1 v1] = GEKriging_predictor(xnod,GEKriging_Model);
% [y2 v2]= GEKriging_predictor(xnod,GEKriging_Model1);
% [y3 v3]= GEKriging_predictor(xnod,GEKriging_Model2);
% 
% Z = reshape(yy,nn,nn); 
% subplot(1,2,1)
% mesh(X,Y,Z'); 
% xlabel('x1'); ylabel('x2'); zlabel('y'); 
% subplot(1,2,2)
% contourf(X,Y,Z',20); hold on
% xlabel('x1'); ylabel('x2'); 
% 
% 
% Z1 = reshape(y1,nn,nn); 
% Z2 = reshape(y2,nn,nn); 
% Z3 = reshape(y3,nn,nn); 
% 
% figure
% mesh(X,Y,Z1');hold on   
% % contourf(X,Y,Z1',20); hold on
% plot3(x(:,1),x(:,2),y,'ro','linewidth',1.5);
% xlabel('x1'); ylabel('x2'); 
% title('GE-Kriging')
% 
% figure
% % contourf(X,Y,Z2',20); hold on
% mesh(X,Y,Z2');hold on
% plot3(x(:,1),x(:,2),y,'ro','linewidth',1.5);
% xlabel('x1'); ylabel('x2'); 
% title('2-appendant SGE-Kriging')
% 
% figure
% mesh(X,Y,Z3'); hold on
% % contourf(X,Y,Z3',20); hold on
% plot3(x(:,1),x(:,2),y,'ro','linewidth',1.5);
% xlabel('x1'); ylabel('x2'); 
% title('3-appendant SGE-Kriging')