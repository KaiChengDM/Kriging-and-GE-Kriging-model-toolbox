%
%  fnormmom.m   Date: 1/25/2016
%  This program computes the mean and covariance matrix of Y=|X|,
%  where X ~ N(mu,S).
%  Input:
%  mu: an nx1 vector of mean of X
%  S: an nxn covariance matrix of X
%  Output
%  muY: E[Y] = E[|X|]
%  varY: Var[Y] = Var[|X|]
%
function [muY,varY] = fnormmom(mu,S)
n = length(mu);
muY = zeros(n,1);
varY = zeros(n);
s = sqrt(diag(S));
h = mu./s;
pdfh = normpdf(h);
cdfh = normcdf(h);
muY = s.*(2*pdfh+h.*erf(h/sqrt(2)));
R = S./(s*s');   % correlation matrix
h1 = h*ones(1,n);
A = (h1-R.*h1')./sqrt(2*(1-R.*R));
A(1:n+1:end) = 0;  % some cleaning up
gam = (h*pdfh').*erf(A);
for i=1:n
    for j=1:n
        if i==j 
           varY(i,i) = mu(i)^2+s(i)^2;
        elseif i>j
           varY(i,j) = varY(j,i);
        else
           r = R(i,j); 
           eta = sqrt(1-r^2);
           p = 4*bnorm(h(i),h(j),r)-2*cdfh(i)-2*cdfh(j)+1;
           c = sqrt(h(i)^2+h(j)^2-2*r*h(i)*h(j))/eta;
           varY(i,j) = s(i)*s(j)*(p*(h(i)*h(j)+r)+2*gam(i,j)+2*gam(j,i)+4*eta/sqrt(2*pi)*normpdf(c));
        end
    end
end
varY = varY-muY*muY';
    