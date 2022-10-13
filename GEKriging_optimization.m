function [x_design objective GEKriging_obj GEKriging_const EI] = GEKriging_optimization(inputpar,hyperpar,subopt) 

% Adaptive GE-Kriging for global optimization

%% Preparation

x          = inputpar.x;
y_obj      = inputpar.y_obj;
grad_y     = inputpar.grad_y;
y_const    = inputpar.y_const;
grad_const = inputpar.grad_const;
lb         = inputpar.lb;
ub         = inputpar.ub;
num_const  = inputpar.num_const;

g          = subopt.objective;
pd_y       = subopt.obj_partial;
const      = subopt.const;
pd_const   = subopt.const_partial;
opt_method = subopt.method;
n_target   = subopt.cost;
acqfun     = subopt.acqfun;

[m n]      = size(x);  

options   = optimoptions('fmincon','Display','off','Algorithm','sqp');  % fmincon options
gaoptions  =  optimoptions('ga','UseParallel', true, 'UseVectorized', false, 'Display','off',...
          'FunctionTolerance',1e-5, 'PopulationSize', 100, 'MaxGenerations', 1000); % GA options
iter      = 0; 

%% Adaptive enrichment of training samples

while (1)

   iter = iter + 1;
   
   %% construct surrogate model
   tic
   inputpar.x = x;
   inputpar.y = y_obj;
   inputpar.grad = grad_y;

   GEKriging_obj =GEKriging_fit(inputpar,hyperpar);         % GE-Kriging model for objective function
 
   for k = 1 : num_const

     inputpar.x = x;
     inputpar.y = y_const{k};
     inputpar.grad = grad_const{k}; 
     GEKriging_const{k} = GEKriging_fit(inputpar,hyperpar);  % GE-Kriging model for constraint function
  
   end   
   toc
 %% Current optimal design
  
 if num_const == 0
    [min_current ind] = min(y_obj);            % current optimal response
    x_design(iter,:)  = x(ind,:);
 else
    for i = 1: num_const
       ind1{i} = find(y_const{i} < 0);
       [minimum(i) ind2(i)] = min(y_obj(ind1{i}));
       xi(i,:)  = x(ind1{i}(ind2(i)),:);
    end 
    [min_current ind] = min(minimum);
    x_design(iter,:)  = xi(ind,:);
 end
    objective(iter) = min_current
 
 %% find the next best training sample point
tic
if num_const == 0
         
         switch acqfun
            case 'EI'
                switch opt_method   
                  case 'GA'
                     for k = 1: 1
                          [x_n(k,:),EI_value(k)] = ga(@(x)Unconst_EI(x,GEKriging_obj,min_current),n,[],[],[],[],lb,ub,[],gaoptions);  % Genetic algorithm
                     end
                  case 'CMAES'
                      for k = 1:5
                           opts.LBounds = lb; opts.UBounds = ub; initial = rand(1,n).*(ub-lb)+lb;
                           [x_n(k,:),EI_value(k)] = Cmaes(@(x)Unconst_EI(x,GEKriging_obj,min_current),initial,[],opts);  % CMAES algorithm
                      end
                  case 'SQP'
                      for k = 1: 5
                           [x_n(k,:),EI_value(k)] = fmincon(@(x)Unconst_EI(x,GEKriging_obj,min_current),rand(1,n).*(ub-lb)+lb,[],[],[],[],lb,ub,[],options); % SQP algorithm for finding best next point
                      end
                 end

            case 'GEI'
                 switch opt_method   
                  case 'GA'
                     for k = 1: 1
                          [x_n(k,:),EI_value(k)] = ga(@(x)Unconst_GEI1(x,GEKriging_obj,min_current),n,[],[],[],[],lb,ub,[],gaoptions);  % Genetic algorithm
                     end
                  case 'CMAES'
                      for k = 1: 5
                           opts.LBounds = lb; opts.UBounds = ub; initial = rand(1,n).*(ub-lb)+lb;
                           [x_n(k,:),EI_value(k)] = Cmaes(@(x)Unconst_GEI1(x,GEKriging_obj,min_current),initial,[],opts);  % CMAES algorithm
                      end
                  case 'SQP'
                      for k = 1: 5
                           [x_n(k,:),EI_value(k)] = fmincon(@(x)Unconst_GEI1(x,GEKriging_obj,min_current),rand(1,n).*(ub-lb)+lb,[],[],[],[],lb,ub,[],options); % SQP algorithm for finding best next point
                      end
                 end
             case 'GPI'
                 switch opt_method   
                  case 'GA'
                     for k = 1: 1
                          [x_n(k,:),EI_value(k)] = ga(@(x)Unconst_GPI(x,GEKriging_obj,min_current),n,[],[],[],[],lb,ub,[],gaoptions);  % Genetic algorithm
                     end
                  case 'CMAES'
                      for k = 1: 5
                           opts.LBounds = lb; opts.UBounds = ub; initial = rand(1,n).*(ub-lb)+lb;
                           [x_n(k,:),EI_value(k)] = Cmaes(@(x)Unconst_GPI(x,GEKriging_obj,min_current),initial,[],opts);  % CMAES algorithm
                      end
                  case 'SQP'
                      for k = 1: 5
                           [x_n(k,:),EI_value(k)] = fmincon(@(x)Unconst_GPI(x,GEKriging_obj,min_current),rand(1,n).*(ub-lb)+lb,[],[],[],[],lb,ub,[],options); % SQP algorithm for finding best next point
                      end
                 end
              case 'PI'
                 switch opt_method   
                  case 'GA'
                     for k = 1: 1
                          [x_n(k,:),EI_value(k)] = ga(@(x)Unconst_PI(x,GEKriging_obj,min_current),n,[],[],[],[],lb,ub,[],gaoptions);  % Genetic algorithm
                     end
                  case 'CMAES'
                      for k = 1: 5
                           opts.LBounds = lb; opts.UBounds = ub; initial = rand(1,n).*(ub-lb)+lb;
                           [x_n(k,:),EI_value(k)] = Cmaes(@(x)Unconst_PI(x,GEKriging_obj,min_current),initial,[],opts);  % CMAES algorithm
                      end
                  case 'SQP'
                      for k = 1: 5
                           [x_n(k,:),EI_value(k)] = fmincon(@(x)Unconst_PI(x,GEKriging_obj,min_current),rand(1,n).*(ub-lb)+lb,[],[],[],[],lb,ub,[],options); % SQP algorithm for finding best next point
                      end
                 end
             case 'LCB'
                [x_n(k,:),EI_value(k)] = Cmaes(@(x)Unconst_LCB(x,GEKriging_obj,min_current),initial,[],opts);  % Find the best next point
        end
    
 else
%     for k = 1: 5
%       [x_next(k,:),EI_value(k)] = fmincon(@(x)Const_EI(x,GEKriging_obj,GEKriging_const,min_current),rand(1,n).*(ub-lb)+lb,[],[],[],[],lb,ub,[],options); % SQP algorithm for finding best next point
      
         switch acqfun
             case 'EI'
                  switch opt_method
                  case 'GA'
                     for k = 1: 1
                          [x_n(k,:),EI_value(k)] = ga(@(x)Const_EI(x,GEKriging_obj,GEKriging_const,min_current),n,[],[],[],[],lb,ub,[],gaoptions);  % Genetic algorithm
                     end
                  case 'CMAES'
                      for k = 1: 5
                           opts.LBounds = lb; opts.UBounds = ub; initial = rand(1,n).*(ub-lb)+lb;
                           [x_n(k,:),EI_value(k)] = Cmaes(@(x)Const_EI(x,GEKriging_obj,GEKriging_const,min_current),initial,[],opts);  % CMAES algorithm
                      end
                  case 'SQP'
                      for k = 1: 5
                           [x_n(k,:),EI_value(k)] = fmincon(@(x)Const_EI(x,GEKriging_obj,GEKriging_const,min_current),rand(1,n).*(ub-lb)+lb,[],[],[],[],lb,ub,[],options); % SQP algorithm for finding best next point
                      end
                  end       
              case 'GEI'
                  switch opt_method
                  case 'GA'
                     for k = 1: 1
                          [x_n(k,:),EI_value(k)] = ga(@(x)Const_GEI(x,GEKriging_obj,GEKriging_const,min_current),n,[],[],[],[],lb,ub,[],gaoptions);  % Genetic algorithm
                     end
                  case 'CMAES'
                      for k = 1: 5
                           opts.LBounds = lb; opts.UBounds = ub; initial = rand(1,n).*(ub-lb)+lb;
                           [x_n(k,:),EI_value(k)] = Cmaes(@(x)Const_GEI(x,GEKriging_obj,GEKriging_const,min_current),initial,[],opts);  % CMAES algorithm
                      end
                  case 'SQP'
                      for k = 1: 5
                           [x_n(k,:),EI_value(k)] = fmincon(@(x)Const_GEI(x,GEKriging_obj,GEKriging_const,min_current),rand(1,n).*(ub-lb)+lb,[],[],[],[],lb,ub,[],options); % SQP algorithm for finding best next point
                      end
                  end       
             
             case 'PI'
                 [x_n(k,:),EI_value(k)] = Cmaes(@(x)Const_PI(x,GEKriging_obj,GEKriging_const,min_current),initial,[],opts);  % Find the best next point
             case 'LCB'
                 [x_n(k,:),EI_value(k)] = Cmaes(@(x)Const_LCB(x,GEKriging_obj,GEKriging_const,min_current),initial,[],opts);  % Find the best next point
         end
  %   end
end
toc
  [value, ind]     = min(EI_value);
  x_next = x_n(ind,:);
  EI(iter) = value;
  %% enrichment of current training sample set for objective function

  x     = [x; x_next];                 % Enrichment of current training samples set
  y_obj = [y_obj; g(x_next)];          % Enrcichment of model response values

  for j = 1:n
     grad_y_new(j) = pd_y{j}(x_next);
  end

  grad_y = [grad_y;grad_y_new];

  %% enrichment of current training sample set for constraint function

  for i = 1 : num_const
     y_const{i} = [y_const{i};const{i}(x_next)];
     for j = 1:n
       grad_const_new(j) = pd_const{i,j}(x_next);
     end
     grad_const{i}   = [grad_const{i}; grad_const_new];
  end

  if iter > n_target
      break;
  end

end
 
 if num_const == 0
   GEKriging_const = [];
 end

end
