function model = SGEKriging_fit_3(inputpar,hyperpar)   

% Training a 3 appendant sliced gradient-enhanced Kriging model

%% Preparation

num            = hyperpar.multistarts;
beta           = hyperpar.beta; 
lb             = hyperpar.lb;
ub             = hyperpar.ub;
x              = inputpar.x;
y              = inputpar.y;
grad_y         = inputpar.grad;
threshold      = inputpar.threshold;
snum           = inputpar.snum;
corr_fun       = hyperpar.corr_fun;  
opt_algorithm  = hyperpar.opt_algorithm;

[m n]       = size(x);                  % number of design sites and their dimension     

lb_input    = min(x);   
ub_input    = max(x); 
mean_output = mean(y); 
std_output  = std(y);

% si          = mean(grad_y.^2);          % Derivative-based global sensitivity indices
% ratio       = si./sum(si);
% [ssi order] = sort(si,'descend');  
% x           = x(:,order);
% grad_y      = grad_y(:,order);

u      = (x-repmat(lb_input,m,1))./(repmat(ub_input,m,1)-repmat(lb_input,m,1));   % Normalization of input data
y      = (y-repmat(mean_output,m,1))./repmat(std_output,m,1);                     % Normalization of output data
grad_f = grad_y.*(ub_input-lb_input)/std_output;                    % Retained partial derivative samples

si          = mean(grad_f.^2);          % Derivative-based global sensitivity indices
ratio       = si./sum(si);
[ssi order] = sort(si,'descend');  
u           = u(:,order);
grad_f      = grad_f(:,order);
yt    = [y grad_f];

for i = 1 :n
  energy(i) = sum(ssi(1:i))./sum(ssi);
end

dim = min(find(energy >= threshold));   % determine the effective dimension 

mn = m*(1+dim);

yt = yt(:,1:dim+1); grad_f = grad_f(:,1:dim);

input_bound   = [lb_input; ub_input];      % Input bound
output_moment = [mean_output; std_output]; % Output moment

model.sensitivity   = si;
model.input_bound   = input_bound;
model.output_moment = output_moment;
model.tran_input    = u;
model.output        = y;
model.orig_grad_output = grad_y;
model.tran_grad_output = grad_f;
model.order = order;
model.orig_input  = x;
model.input_dim   = n;
model.sample_size = m;
model.dim         = dim;
model.corr_fun    = corr_fun; 
model.snum        = snum;

%% Split sample set into m parts
 
 [si_max index] = max(si); 

 [vals, uind] = sort(u(:,index));
 
 Hn = fix(m/snum);  re = mod(m,snum);
 
 if  re == 0
    for i = 1 : snum
         ind{i} = uind((i-1)*Hn+1:i*Hn);   
    end
 else
    for i = 1 : snum
        if i <= re
            ind{i} = uind((i-1)*(Hn+1)+1:i*(Hn+1));
        else
            ind{i} = uind(re*(Hn+1)+(i-re-1)*Hn+1:re*(Hn+1)+(i-re)*Hn);
        end  
    end
 end
 
 for i = 1 : snum - 2

     ind3{i} = sort([ind{i};ind{i+1};ind{i+2}]);
     ind2{i} = sort([ind{i+1};ind{i+2}]);

     u3{i} = u(ind3{i},:); n3 = length(ind3{i});
     u2{i} = u(ind2{i},:); n2 = length(ind2{i});
     
     y3{i} = reshape(yt(ind3{i},1:dim+1),n3*(dim+1),1); 
     y2{i} = reshape(yt(ind2{i},1:dim+1),n2*(dim+1),1);
    
     f3{i} = [ones(n3,1); zeros(n3*dim,1)];  
     f2{i} = [ones(n2,1); zeros(n2*dim,1)];  

 end

%%  Hyper-parameters tuning

log_ub = log10(ub); log_lb = log10(lb); 

for kk = 1: num                                              % Multistart for hype-parameter optimization

 beta = 10.^(rand(1,3).*(log_ub-log_lb)+log_lb);
% beta = 10.^(rand.*ones(1,3).*(log_ub-log_lb)+log_lb);

  switch opt_algorithm   

     case 'Hooke-Jeeves'
          [t_opt(kk,:),fmin(kk), perf] = boxmin(beta, lb, ub, model);  % Hooke & Jeeves Algorithm 
     case 'Fmincon'
          options = optimoptions('fmincon','Display','off','Algorithm','sqp'); % fmincon
          [t_opt(kk,:),fmin(kk)] = fmincon(@(x)GEKriging_likelihood(x),beta,[],[],[],[],lb,ub,[],options);
     case 'GA'
          gaoptions  =  optimoptions('ga','UseParallel', true, 'UseVectorized', false, 'Display','off',...
          'FunctionTolerance',1e-3, 'PopulationSize', 800, 'MaxGenerations', 2000);
          [t_opt(kk,:),fmin(kk)] = ga(@(x)GEKriging_likelihood(x),size(beta,2),[],[],[],[],lb,ub,[],gaoptions);% Genetic algorithm
     case 'CMAES'
          opts.LBounds = lb; opts.UBounds = ub; 
          [t_opt(kk,:),fmin(kk)] = Cmaes(@(x)GEKriging_likelihood(x),beta,[],opts);
    end
end
%  t_opt
%  fmin
 
 [value, ind]     = min(fmin);
 beta             = t_opt(ind,:);
 model.beta       = beta;
 model.likelihood = value;
 theta    = beta(1).*ratio.^beta(2)+beta(3);
 corrmat  = feval(corr_fun,u,theta,dim,'on');
 
 [upper_mat rd] = chol(corrmat);        %  Matrix inversion with cholesky
 model.upper_mat = upper_mat;
 model.corrmat   = corrmat;
 
 f  = [ones(m,1); zeros(m*dim,1)]; yt = reshape(yt,m*(dim+1),1);
 beta0  = f'*(upper_mat\(upper_mat'\yt))/sum((upper_mat\f).^2);
 sigma2 = sum((upper_mat'\(yt-beta0*f)).^2)/mn;
 
 model.theta  = theta;
 model.beta0  = beta0;
 model.sigma2 = sigma2;
 
%% likelihood function 
 
function Likelihood = GEKriging_likelihood(beta) 
   
% theta = beta(1).*(ratio+beta(2)).^beta(3)+beta(4);

 theta = beta(1).*ratio.^beta(2)+beta(3);

 for i = 1 : snum-2
     
     sub_mat3 = feval(corr_fun,u3{i},theta,dim,'on');

     [upper_mat3{i} rd3] = chol(sub_mat3);  
     
%    det3(i) = prod(diag(upper_mat3{i}).^(2/mn));  

     logdet3(i) = 2*sum(log(diag(upper_mat3{i}))); 

     nom3(i) = f3{i}'*(upper_mat3{i}\(upper_mat3{i}'\y3{i})); 
     
     denom3(i) = sum((upper_mat3{i}\f3{i}).^2);
 end

 for i = 1 : snum-3

      sub_mat2 = feval(corr_fun,u2{i},theta,dim,'on');

      [upper_mat2{i} rd2] = chol(sub_mat2); 
         
%     det2(i) = prod(diag(upper_mat2{i}).^(2/mn)); 

      logdet2(i) = 2*sum(log(diag(upper_mat2{i}))); 

      nom2(i) = f2{i}'*(upper_mat2{i}\(upper_mat2{i}'\y2{i})); 
         
      denom2(i) = sum((upper_mat2{i}\f2{i}).^2);
  
 end

 beta = (sum(nom3)-sum(nom2))/(sum(denom3)-sum(denom2));
   
 for i = 1 : snum-3
     sig3(i) = sum((upper_mat3{i}'\(y3{i}-beta*f3{i})).^2); 
     sig2(i) = sum((upper_mat2{i}'\(y2{i}-beta*f2{i})).^2); 
 end

 sig3(snum-2) = sum((upper_mat3{snum-2}'\(y3{snum-2}-beta*f3{snum-2})).^2); 

 sigma2 = (sum(sig3)-sum(sig2))/mn; 
  
%  Likelihood = sigma2*prod(det3)/prod(det2);
 Likelihood = mn*log(sigma2)+sum(logdet3)-sum(logdet2);
end

%% Hooke & Jeeves Algorithm

function  [t, f, perf]  =  boxmin(t0, lo, up,par)
    
% Hooke & Jeeves Algorithm for hyper-parameters optimization (This part comes from Dace toolbox)

% Initialize
[t, f, itpar]  =  start(t0, lo, up, par);
if  ~isinf(f)
  % Iterate
  p  =  length(t);
  if  p <=  2,  kmax  =  2; else,  kmax  =  min(p,4); end
  for  k  =  1 : kmax
    th  =  t;
    [t, f,  itpar]  =  explore(t, f,  itpar, par);
    [t, f,  itpar]  =  move(th, t, f, itpar, par);
  end
end
perf  =  struct('nv',itpar.nv, 'perf',itpar.perf(:,1:itpar.nv));
end


function  [t, f, itpar]  =  start(t0, lo, up, par)
% Get starting point and iteration parameters

% Initialize
t  =  t0(:);  lo  =  lo(:);   up  =  up(:);   p  =  length(t);
D  =  2 .^ ([1:p]'/(p+2));
ee  =  find(up  ==  lo);  % Equality constraints
if  ~isempty(ee)
  D(ee)  =  ones(length(ee),1);   t(ee)  =  up(ee); 
end
ng  =  find(t < lo | up < t);  % Free starting values
if  ~isempty(ng)
  t(ng)  =  (lo(ng) .* up(ng).^7).^(1/8);  % Starting point
end
ne  =  find(D ~=  1);

% Check starting point and initialize performance info
[f]  =  GEKriging_likelihood(t);   nv  =  1;
itpar  =  struct('D',D, 'ne',ne, 'lo',lo, 'up',up, ...
  'perf',zeros(p+2,200*p), 'nv',1);
itpar.perf(:,1)  =  [t; f; 1];
if  isinf(f)    % Bad parameter region
  return
end

if  length(ng) > 1  % Try to improve starting guess
  d0  =  16;  d1  =  2;   q  =  length(ng);
  th  =  t;   fh  =  f;   jdom  =  ng(1);  
  for  k  =  1 : q
    j  =  ng(k);    fk  =  fh;  tk  =  th;
    DD  =  ones(p,1);  DD(ng)  =  repmat(1/d1,q,1);  DD(j)  =  1/d0;
    alpha  =  min(log(lo(ng) ./ th(ng)) ./ log(DD(ng))) / 5;
    v  =  DD .^ alpha;   tk  =  th;
    for  rept  =  1 : 4
      tt  =  tk .* v; 
      [ff ]  =  GEKriging_likelihood(tt);  nv  =  nv+1;
      itpar.perf(:,nv)  =  [tt; ff; 1];
      if  ff <=  fk 
        tk  =  tt;  fk  =  ff;
        if  ff <=  f
          t  =  tt;  f  =  ff;  jdom  =  j;
        end
      else
        itpar.perf(end,nv)  =  -1;   break
      end
    end
  end % improve
  
  % Update Delta  
  if  jdom > 1
    D([1 jdom])  =  D([jdom 1]); 
    itpar.D  =  D;
  end
end % free variables

itpar.nv  =  nv;
end
% --------------------------------------------------------

function  [t, f, itpar]  =  explore(t, f, itpar, par)
% Explore step

nv  =  itpar.nv;   ne  =  itpar.ne;
for  k  =  1 : length(ne)
  j  =  ne(k);   tt  =  t;   DD  =  itpar.D(j);
  if  t(j)  ==  itpar.up(j)
    atbd  =  1;   tt(j)  =  t(j) / sqrt(DD);
  elseif  t(j)  ==  itpar.lo(j)
    atbd  =  1;  tt(j)  =  t(j) * sqrt(DD);
  else
    atbd  =  0;  tt(j)  =  min(itpar.up(j), t(j)*DD);
  end
%   [ff  fitt]  =  objfunc(tt,par);  nv  =  nv+1;
[ff]  =  GEKriging_likelihood(tt);  nv  =  nv+1;
  itpar.perf(:,nv)  =  [tt; ff; 2];
  if  ff < f
    t  =  tt;  f  =  ff; 
  else
    itpar.perf(end,nv)  =  -2;
    if  ~atbd  % try decrease
      tt(j)  =  max(itpar.lo(j), t(j)/DD);
%       [ff  fitt]  =  objfunc(tt,par);  nv  =  nv+1;
        [ff ]  = GEKriging_likelihood(tt);  nv  =  nv+1;   
      itpar.perf(:,nv)  =  [tt; ff; 2];
      if  ff < f
        t  =  tt;  f  =  ff; 
      else
        itpar.perf(end,nv)  =  -2;
      end
    end
  end
end % k

itpar.nv  =  nv;
end
% --------------------------------------------------------

function  [t, f, itpar]  =  move(th, t, f, itpar, par)
% Pattern move

nv  =  itpar.nv;   ne  =  itpar.ne;   p  =  length(t);
v  =  t ./ th;
if  all(v  ==  1)
  itpar.D  =  itpar.D([2:p 1]).^.2;
  return
end

% Proper move
rept  =  1;
while  rept
  tt  =  min(itpar.up, max(itpar.lo, t .* v));  
%   [ff  fitt]  =  objfunc(tt,par);  nv  =  nv+1;
  [ff ]  =  GEKriging_likelihood(tt);  nv  =  nv+1;
  itpar.perf(:,nv)  =  [tt; ff; 3];
  if  ff < f
    t  =  tt;  f  =  ff;  
    v  =  v .^ 2;
  else
    itpar.perf(end,nv)  =  -3;
    rept  =  0;
  end
  if  any(tt == itpar.lo | tt == itpar.up), rept  =  0; end
end

itpar.nv  =  nv;
itpar.D  =  itpar.D([2:p 1]).^.25;
end  
end   

