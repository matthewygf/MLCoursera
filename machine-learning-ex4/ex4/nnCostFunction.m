function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

input = X;
% add ones to input
a1 = [ones(m, 1), input];
inputLayerOutput = sigmoid(a1 * Theta1');
a2 = [ones(m, 1) inputLayerOutput];

h = sigmoid(a2 * Theta2');

% Constructing a vector if x (subscript i) = digit 5, then yVec = 
% fifth position [0 0 0 0 1 0 0 0 0 0] where rows are training set samples
% since y is all out training data. we want to get a 5000 rows 10 cols vector.
% that each row is just zeros or one, by getting this from 1 - 10 labels.
trainsetCopiesOfLabels = repmat([1:num_labels], m, 1); % 5000 x 10
yCopies = repmat(y, 1 , num_labels);
yClassVec = trainsetCopiesOfLabels == yCopies; %5000 x 10

logH = log(h);
left = -yClassVec .* logH;
right = (1 - yClassVec) .* (log(1-h)) ;

cost = left - right;

% sum the number of labels / class
% sum the number of inputs  
J = (1 / m) * sum(sum(cost));

% Problem 2 solution
theta1ExcludingBias = Theta1(:, 2:end);
theta2ExcludingBias = Theta2(:, 2:end);

reg1 = sum(sum(theta1ExcludingBias .^ 2));
reg2 = sum(sum(theta2ExcludingBias .^ 2));
regularizedTerm = (lambda / (2 * m)) * (reg1 + reg2);
J = (1 / m) * sum(sum(cost)) + regularizedTerm;

delta1 = zeros(size(Theta1));
delta2 = zeros(size(Theta2));

for t = 1:m,

	ht = h(t, :)';
	a1t = a1(t,:)';
	a2t = a2(t, :)';
	yVect = yClassVec(t, :)';

	d3t = ht - yVect;
	z2t = [1; Theta1 * a1t];
  d2t = Theta2' * d3t .* sigmoidGradient(z2t);

  delta1 = delta1 + d2t(2:end) * a1t';
  delta2 = delta2 + d3t * a2t';
end;

Theta1_grad = (1 / m) * delta1;
Theta2_grad = (1 / m) * delta2;


% we want to set Theta's first column to zero ,
% so that we do not regularize for the bias term.
regularizedTermOne = lambda / m * [zeros(size(Theta1, 1), 1) theta1ExcludingBias];
Theta1_grad = (1 / m) * delta1 + regularizedTermOne;

regularizedTermTwo = lambda / m * [zeros(size(Theta2, 1),1) theta2ExcludingBias];
Theta2_grad = (1 / m) * delta2 + regularizedTermTwo;













% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
