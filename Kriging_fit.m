function model = Kriging_fit(inputpar,hyperpar)   

% Training a Kriging model with constant mean

%%  Preparation

theta    = hyperpar.theta ; 
lb       = hyperpar.lb;
ub       = hyperpar.ub;
num      = hyperpar.multistarts;
corr_fun = hyperpar.corr_fun;
opt_algorithm = hyperpar.opt_algorithm;

x      = inputpar.x;
y      = inputpar.y;

[m, n] = size(x);               % number of design sites and their dimension     

dim = n;  mn = m; 

lb_input    = min(x);   
ub_input    = max(x); 
mean_output = mean(y); 
std_output  = std(y);

u = (x-repmat(lb_input,m,1))./(repmat(ub_input,m,1)-repmat(lb_input,m,1));   % Normalization of input data
y = (y-repmat(mean_output,m,1))./repmat(std_output,m,1);  % Normalization of output data
f = ones(m,1);   
 
nugget = 10^-10*eye(m);

input_bound   = [lb_input; ub_input];  % Input bound
output_moment = [mean_output; std_output]; % Output moment

model.tran_input       = u;
model.output           = y;
model.orig_input       = x;
model.input_dim        = n;
model.sample_size      = m;
model.dim              = dim;
model.order            = 1:n;
model.corr_fun         = corr_fun;
model.input_bound      = input_bound;
model.output_moment    = output_moment;

%%  Hyper-parameter optimization

log_ub = log10(ub); log_lb = log10(lb); 

for kk = 1: num                                  % Multistart for hype-parameter optimization

 theta = 10.^(rand.*ones(1,n).*(log_ub-log_lb)+log_lb);

 switch opt_algorithm   
   
   case 'Hooke-Jeeves'
         [t_opt(kk,:), fmin(kk), perf] = boxmin(theta, lb, ub, model);  % Hooke & Jeeves Algorithm 
   case 'Fmincon'
        options = optimoptions('fmincon','Display','off','Algorithm','sqp'); % fmincon
        [t_opt(kk,:),fmin(kk)] = fmincon(@(x)GEKriging_likelihood(x),theta,[],[],[],[],lb,ub,[],options);
   case 'GA'
        gaoptions  =  optimoptions('ga','UseParallel', true, 'UseVectorized', false, 'Display','off',...
        'FunctionTolerance',1e-3, 'PopulationSize', 800, 'MaxGenerations', 2000);
        [t_opt(kk,:),fmin(kk)] = ga(@(x)GEKriging_likelihood(x),size(theta,2),[],[],[],[],lb,ub,[],gaoptions);% Genetic algorithm
   case 'CMAES'
        opts.LBounds = lb; opts.UBounds = ub; 
        [t_opt(kk,:),fmin(kk)] = Cmaes(@(x)GEKriging_likelihood(x),theta,[],opts);
  end

end

 [value, ind]     = min(fmin);
 model.theta      = t_opt(ind,:);
 model.likelihood = value;
 
 corrmat  = feval(corr_fun,u,model.theta,dim,'off');

 [C, rd]  = chol(corrmat+nugget);  CT = C';      % Full correlation matrix 

 model.upper_mat = C;
 model.corrmat   = corrmat;

 beta0  = f'*(C\(CT\y))/sum((CT\f).^2);
 sigma2 = sum((CT\(y-beta0*f)).^2)/mn;

 model.beta0  = beta0;
 model.sigma2 = sigma2;

 %% Likelihood function

function Likelihood = Kriging_likelihood(theta) 

  corrmat  = feval(corr_fun,u,theta,dim,'off');

  [C, rd] = chol(corrmat+nugget); CT = C';
                        
  beta0 = f'*(C\(CT\y))/sum((CT\f).^2);
  
  sigma2 = sum((CT\(y-beta0*f)).^2)/mn;

  % detR = prod(diag(upper_mat).^(2/mn));  

  % Likelihood = sigma2*detR;  % likelihood function of weighted krging model
  
  Likelihood = mn*log(sigma2)+2*sum(log(diag(C)));

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
[f]  =  Kriging_likelihood(t);   nv  =  1;
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
      [ff ]  =  Kriging_likelihood(tt);  nv  =  nv+1;
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
[ff]  =  Kriging_likelihood(tt);  nv  =  nv+1;
  itpar.perf(:,nv)  =  [tt; ff; 2];
  if  ff < f
    t  =  tt;  f  =  ff; 
  else
    itpar.perf(end,nv)  =  -2;
    if  ~atbd  % try decrease
      tt(j)  =  max(itpar.lo(j), t(j)/DD);
%       [ff  fitt]  =  objfunc(tt,par);  nv  =  nv+1;
        [ff ]  = Kriging_likelihood(tt);  nv  =  nv+1;   
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
  [ff ]  =  Kriging_likelihood(tt);  nv  =  nv+1;
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

