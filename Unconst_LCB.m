function LCB = Unconst_LCB(x,model,fmin)  

% Compute the expected improvement function 

  [mean,varaince] = GEKriging_predictor(x,model);
 
  std = sqrt(varaince);

  f = (fmin-mean)./std;

  LCB = mean-4*std;

end
