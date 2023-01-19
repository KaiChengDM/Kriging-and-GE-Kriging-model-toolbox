function [pred_mean pred_variance  pred_covariance] = Kriging_predictor(x_pre,model)  

% GE-Kriging model predictor

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
         corrmat  =  corrgaussian(u_pre,theta,dim,'off');   
       case 'corrspline'
         corrvector = Spline_corrvector(u_pre,model,'off');      % Correlation vector
         corrmat  =  corrspline(u_pre,theta,dim,'off');   
       case 'corrbiquadspline'
         corrvector = Biquadspline_corrvector(u_pre,model,'off'); % correlation matrix
         corrmat   =  corrbiquadspline(u_pre,theta,dim,'off');   
 end  

 %% Prediction

 upper_mat = model.upper_mat;

 f = ones(m,1);                    

 beta0 = model.beta0;   sigma2 = model.sigma2;

 mean = beta0+corrvector*(upper_mat\(upper_mat'\(y-f*beta0))); % Kriging prediction mean

 u = (corrvector*(upper_mat\(upper_mat'\f))-1)/(upper_mat\f);

 rt = upper_mat'\corrvector'; 

 variance =  sigma2*(1- sum(rt.^2) + sum(u'.^2))';    % prediction variance

 full_corrmat = corrmat+corrmat'-eye(n1);

 covariance = sigma2*(full_corrmat-rt'*rt+u*u');           % prediction covariance 

 output_moment = model.output_moment; 
 pred_mean     = mean.*output_moment(2)+output_moment(1);    % Original prediction mean
 pred_variance = variance.*output_moment(2).^2;              % Original prediction variance
 pred_covariance = covariance.*output_moment(2).^2;          % Original prediction covariance

end
