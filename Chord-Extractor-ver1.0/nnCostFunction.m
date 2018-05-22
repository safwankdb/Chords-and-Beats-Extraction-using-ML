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

X=[ones(size(X),1) X];
         
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
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



for i=1:m

temp=zeros(num_labels,1);
temp(y(i))=1;

x=X(i,:)';

a1=sigmoid(Theta1*x);
a1=[1; a1];
h=sigmoid(Theta2*a1);

J=J+sum(-temp.*log(h)-(1-temp).*log(1-h));
end

J=J/m;
Theta1p=Theta1;
Theta2p=Theta2;
Theta1p(:,1)=zeros(size(Theta1p,1),1);
Theta2p(:,1)=zeros(size(Theta2p,1),1);
J=J+ ((lambda*0.5)/m)*((sum(sum(Theta1p.^2)))+sum(sum(Theta2p.^2)));

Delta_1=zeros(size(Theta1));
Delta_2=zeros(size(Theta2));

for t=1:m

a1=X(t,:)';
yf=zeros(num_labels,1);
yf(y(t))=1;

z2=Theta1*a1;
%z2=[1; z2];
a2=sigmoid(z2);
a2=[1;a2];
z3=Theta2*a2;
a3=sigmoid(z3);

z2=[100;z2];     %% very very important you can put anything in place of 100 doesn't matter'.
delta3=a3-yf;

delta2=(Theta2'*delta3).*sigmoidGradient(z2);
        
        delta2=delta2(2:end);
        
        %delta1=(Theta1'*delta2).*sigmoidGradient(z1);
                
         %       delta1=delta1(2:end);
                
                
                Delta_1=Delta_1+delta2*a1';
                Delta_2=Delta_2+delta3*a2';
                
                end
                
                 
                 Theta1(:,1)=zeros(size(Theta1,1),1);
                 Theta2(:,1)=zeros(size(Theta2,1),1);
                 
                Theta1_grad=(Delta_1+lambda*Theta1)/m;
                Theta2_grad=(Delta_2+lambda*Theta2)/m;
                
                
                
























% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
