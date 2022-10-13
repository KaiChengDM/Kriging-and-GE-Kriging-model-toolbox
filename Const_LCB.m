function LCB = Const_LCB(x,obj_model,const_model,fmin)  

% Compute the expected improvement function 

  [obj_mean,obj_varaince] = GEKriging_predictor(x,obj_model);

  obj_std = sqrt(obj_varaince); 

  num = size(const_model,2);  
  
  for i = 1:num
     [const_mean(i,:),varaince(i,:)] = GEKriging_predictor(x,const_model{i});
     prob(i,:) = normcdf(-const_mean(i,:)./sqrt(varaince(i,:)));
  end
 
  LCB = (obj_mean-4*obj_std).*prod(prob);

end
