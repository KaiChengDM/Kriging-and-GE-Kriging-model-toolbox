function PI = Unconst_PI(x,model,fmin)  

% Compute the probability of improvement

  [mean,varaince] = GEKriging_predictor(x,model);
 
  PI = -normcdf((fmin-mean)./sqrt(varaince));

end
