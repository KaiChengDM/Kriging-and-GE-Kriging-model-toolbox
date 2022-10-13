function [pred_mean pred_variance] = Kriging_predictor(x_pre,model)  

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
         corrvector = Gaussian_corrvector(u_pre,model,'off'); % Correlation vector 
       case 'corrspline'
         corrvector = Spline_corrvector(u_pre,model,'off'); % Correlation vector
       case 'corrbiquadspline'
         corrvector = Biquadspline_corrvector(u_pre,model,'off'); % Reduced correlation matrix
 end  

 %% Prediction

 upper_mat = model.upper_mat;

 f = ones(m,1);                    

 beta0 = model.beta0;   sigma2 = model.sigma2;

 mean = beta0+corrvector*(upper_mat\(upper_mat'\(y-f*beta0))); % GE-Kriging prediction mean

 u = (corrvector*(upper_mat\(upper_mat'\f))-1)/(upper_mat\f);

 rt = upper_mat'\corrvector'; 

 variance =  sigma2*(1- sum(rt.^2) + sum(u'.^2))';

 output_moment = model.output_moment; 
 pred_mean     = mean.*output_moment(2)+output_moment(1);    % Original prediction mean
 pred_variance = variance.*output_moment(2).^2;          % Original prediction variance

end
