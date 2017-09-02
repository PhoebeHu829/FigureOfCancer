clear all; close all; clc

load BreastCancerSmall.mat

b = BreastCancerSmall(:, end);  % 0 means benign; 1 means malignant
A = BreastCancerSmall(:, 1:9); % first columns are the predictors 

% normalize columns of A 
for i = 1:9
    A(:,i) = A(:,i)/norm(A(:,i));
end

lambda = .1;
tolerance = 1e-5; 

[m,n] = size(A);

%L = norm(A(:))^2; 
L = norm(A)^2;
converged = 0;

x = zeros(n,1);
iter = 0;
while(~converged)
iter = iter +1;   
[f,g] = myLogistic(x, A, b, lambda);    
x = x - g/L;     
converged  = norm(g) < tolerance;
fprintf('iter: %d, value: %7.3f, norm g: %7.3e\n', iter, f, norm(g));

end


