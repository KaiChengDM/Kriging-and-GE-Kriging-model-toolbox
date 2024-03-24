function [pred_mean, pred_variance] = Kriging_predictor(x_pre,model)  

% Kriging model predictor with constant mean

%% Preparation

input_bound = model.input_bound;  

corr_fun = model.corr_fun;  n1 = size(x_pre,1);  

u_pre = (x_pre-repmat(input_bound(1,:),n1,1))./(repmat(input_bound(2,:),n1,1)-repmat(input_bound(1,:),n1,1)); % Normalization

theta = model.theta;   
y = model.output; 
m = model.sample_size;  
dim = model.dim; 
  
switch corr_fun             
       case 'corrgaussian'
         corrvector = Gaussian_corrvector(u_pre,model,'off');    % Correlation vector 
         % corrmat  =  corrgaussian(u_pre,theta,dim,'off');   
       case 'corrspline'
         corrvector = Spline_corrvector(u_pre,model,'off');      % Correlation vector
         % corrmat  =  corrspline(u_pre,theta,dim,'off');   
       case 'corrbiquadspline'
         corrvector = Biquadspline_corrvector(u_pre,model,'off'); % correlation matrix
         % corrmat   =  corrbiquadspline(u_pre,theta,dim,'off');   
       case 'corrmatern'
            corrvector = Matern_corrvector(u_pre,model,'off'); % correlation matrix
 end  

 %% Prediction

 C = model.upper_mat; CT = C';

 f = ones(m,1);                    

 beta0 = model.beta0;   sigma2 = model.sigma2;

 mean = beta0+corrvector*(C\(CT\(y-f*beta0))); % Kriging prediction mean

 u = (corrvector*(C\(CT\f))-1)/(CT\f);

 rt = CT\corrvector'; 

 variance =  sigma2*(1- sum(rt.^2) + sum(u'.^2))';    % prediction variance

 % full_corrmat = corrmat+corrmat'-eye(n1);

 output_moment = model.output_moment; 
 pred_mean     = mean.*output_moment(2)+output_moment(1);    % Original prediction mean
 pred_variance = variance.*output_moment(2).^2;              % Original prediction variance

end
