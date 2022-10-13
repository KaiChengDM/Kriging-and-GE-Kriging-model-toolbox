clc;  clear;

format long;

hyperpar = [1 1 3];

for k = 3
    
dist = -1:0.01:1;  theta = hyperpar(k);

xi = theta.*abs(dist);

ss_cubic = zeros(1,length(xi));

i1 = find(xi <= 0.2);
i2 = find(0.2 < xi & xi < 1);
if  ~isempty(i1)
  ss_cubic(i1) = 1 - xi(i1).^2 .* (15  - 30*xi(i1));
end
if  ~isempty(i2)
  ss_cubic(i2) = 1.25 * (1 - xi(i2)).^3;
end

ss_biquad = zeros(1,length(xi));
i1 = find(xi <= 0.4);
i2 = find(0.4 < xi & xi < 1);
if  ~isempty(i1)
  ss_biquad(i1) = 1 - 15*xi(i1).^2 + 35*xi(i1).^3 - 195/8*xi(i1).^4;
end
if  ~isempty(i2)
  ss_biquad(i2) = 5/3 - 20/3*xi(i2) + 10*xi(i2).^2 - 20/3*xi(i2).^3 + 5/3*xi(i2).^4;
end

ss_gaussian = exp(-theta^2.*abs(dist).^2);

% if k == 1 
  plot(dist,ss_cubic,'m-','linewidth',2); hold on
% elseif k == 2 
%   plot(dist,ss_cubic,'m-','linewidth',2); hold on
% else
%   plot(dist,ss_cubic,'m-','linewidth',2); hold on
% end

% if k == 1 
  plot(dist,ss_biquad,'b-.','linewidth',2); hold on
% elseif k == 2 
%   plot(dist,ss_biquad,'b-.','linewidth',2); hold on
% else
%   plot(dist,ss_biquad,'b-.','linewidth',2); hold on
% end

  plot(dist,ss_gaussian,'r--','linewidth',2); hold on

%   ss_mat = (1+sqrt(3).*xi).*exp(-sqrt(3).*xi);
%   plot(dist,ss_mat,'k-','linewidth',2); hold on

end

xlabel('Distance'); ylabel('Correlation function value'); 

legend('cubic','biquad','gaussian')

% legend('\theta=0.5','\theta=1','\theta=5')