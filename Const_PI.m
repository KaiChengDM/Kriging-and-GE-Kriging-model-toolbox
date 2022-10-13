function PI = Const_PI(x,obj_model,const_model,fmin)  

% Compute the probability of improvement

  [obj_mean,obj_varaince] = GEKriging_predictor(x,obj_model);

  obj_std = sqrt(obj_varaince); f = (fmin-obj_mean)./obj_std;

  num = size(const_model,2);
  
  for i = 1:num
     [const_mean(i,:),varaince(i,:)] = GEKriging_predictor(x,const_model{i});
     prob(i,:) = normcdf(-const_mean(i,:)./sqrt(varaince(i,:)));
  end


 PI = -(normcdf(f).*prod(prob));

end
