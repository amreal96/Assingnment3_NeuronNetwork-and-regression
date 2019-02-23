function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
%% Compute cost function
htheta = sigmoid(X * theta);

J = (1/m)* sum((-y'*log(htheta)) - (1-y)' * log(1 - htheta))+(lambda/(2*m))*sum(theta(2:end,1).^2);

%Gradient 


grad(1,1) = (1/m)*sum((htheta-y).*X(:,1)); 


grad(2:end,1)=((1/m)*((htheta-y)'*X(:,2:end)))'+(lambda/m)*theta(2:end);







% =============================================================

grad = grad(:);

end
