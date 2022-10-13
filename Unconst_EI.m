function EI = Unconst_EI(x,model,fmin)  

% Compute the expected improvement function 

  [mean,varaince] = GEKriging_predictor(x,model);
 
  std = sqrt(varaince);

  f = (fmin-mean)./std;
 
  EI = -((fmin-mean).*normcdf(f)+std.*normpdf(f));

  EI(find(std==0)) = 0;
  
%   if std > 0
%       EI = -((fmin-mean).*normcdf(f)+std.*normpdf(f));
%   else 
%       EI = 0;
%   end

end
