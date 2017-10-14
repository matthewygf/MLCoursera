function [lambda_vec, error_train, error_val, error_test] = ...
    validationCurve(X, y, Xval, yval, Xtest, ytest)
%VALIDATIONCURVE Generate the train and validation errors needed to
%plot a validation curve that we can use to select lambda
%   [lambda_vec, error_train, error_val] = ...
%       VALIDATIONCURVE(X, y, Xval, yval) returns the train
%       and validation errors (in error_train, error_val)
%       for different values of lambda. You are given the training set (X,
%       y) and validation set (Xval, yval).
%

% Selected values of lambda (you should not change this)
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';

% You need to return these variables correctly.
error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);
error_test = zeros(length(lambda_vec), 1);
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the validation errors in error_val. The 
%               vector lambda_vec contains the different lambda parameters 
%               to use for each calculation of the errors, i.e, 
%               error_train(i), and error_val(i) should give 
%               you the errors obtained after training with 
%               lambda = lambda_vec(i)
%
% Note: You can loop over lambda_vec with the following:
%
%       for i = 1:length(lambda_vec)
%           lambda = lambda_vec(i);
%           % Compute train / val errors when training linear 
%           % regression with regularization parameter lambda
%           % You should store the result in error_train(i)
%           % and error_val(i)
%           ....
%           
%       end
%
%

m = size(X, 1);

for i = 1:m
  for j = 1: length(lambda_vec)
    lambda = lambda_vec(j);
    % use training set
    xTrainSet = X(1:i, :);
    yTrainSet = y(1:i);
    
    % get learned theta
    thetaTrained = trainLinearReg(xTrainSet, yTrainSet, lambda);
    % compute error on training set.
    [jTrainError, trainGrad] = linearRegCostFunction(xTrainSet, yTrainSet, thetaTrained, 0);
    % entire set for cross validation testing
    [jCrossError, crossGrad] = linearRegCostFunction(Xval, yval, thetaTrained, 0);
    [jTestError, TestGrad] = linearRegCostFunction(Xtest, ytest, thetaTrained, 0);
    % store error for each set
    error_train(j) = jTrainError;
    error_val(j) = jCrossError;
    error_test(j) = jTestError;
  end
end







% =========================================================================

end
