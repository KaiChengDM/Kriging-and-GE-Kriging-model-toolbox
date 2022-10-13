function EI = Const_EI(x,obj_model,const_model,fmin)  

% Compute the expected improvement function 

  [obj_mean,obj_varaince] = GEKriging_predictor(x,obj_model);

  obj_std = sqrt(obj_varaince); f = (fmin-obj_mean)./obj_std;

  num = size(const_model,2);
  
  for i = 1:num
     [const_mean(i,:),varaince(i,:)] = GEKriging_predictor(x,const_model{i});
     prob(i,:) = normcdf(-const_mean(i,:)./sqrt(varaince(i,:)));
  end

%   if obj_std>0
%       EI = -((fmin-obj_mean).*normcdf(f)+obj_std.*normpdf(f)).*prod(prob);
%   else 
%       EI = 0;
%   end

  EI = -((fmin-obj_mean).*normcdf(f)+obj_std.*normpdf(f)).*prod(prob);

  EI(find(obj_std==0)) = 0;
end
