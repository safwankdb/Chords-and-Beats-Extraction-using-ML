
clear ; close all; clc

%% Setup the parameters you will use for this exercise
input_layer_size  = 12;
hidden_layer_size = 35;
num_labels = 10;

X=rand(4000,12);
y=(floor(rand(4000,1)*10))+1;
Theta1=rand(35,13);
Theta2=rand(10,36);
m = size(X, 1);
nn_params = [Theta1(:) ; Theta2(:)];



% Weight regularization parameter (we set this to 0 here).
lambda = 0;
%J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
%                   num_labels, X, y, lambda);
%fprintf(['Cost at parameters (loaded from ex4weights): %f '...
    %'\n(this value should be about 0.287629)\n'], J);
%fprintf('\nProgram paused. Press enter to continue.\n');
%pause;
%% =============== Part 4: Implement Regularization ===============
%  Once your cost function implementation is correct, you should now
%  continue to implement the regularization with the cost.
%fprintf('\nChecking Cost Function (w/ Regularization) ... \n')
% Weight regularization parameter (we set this to 1 here).
lambda = 1;
%J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
 %                  num_labels, X, y, lambda);
%fprintf(['Cost at parameters (loaded from ex4weights): %f '...
%         '\n(this value should be about 0.383770)\n'], J);
%fprintf('Program paused. Press enter to continue.\n');
%pause;
%% ================ Part 5: Sigmoid Gradient  ================
%  Before you start implementing the neural network, you will first
%  implement the gradient for the sigmoid function. You should complete the
%  code in the sigmoidGradient.m file.
%fprintf('\nEvaluating sigmoid gradient...\n')
%g = sigmoidGradient([-1 -0.5 0 0.5 1]);
%fprintf('Sigmoid gradient evaluated at [-1 -0.5 0 0.5 1]:\n  ');
%fprintf('%f ', g);
%fprintf('\n\n');
%fprintf('Program paused. Press enter to continue.\n');
%pause;
%% ================ Part 6: Initializing Pameters ================
%  In this part of the exercise, you will be starting to implment a two
%  layer neural network that classifies digits. You will start by
%  implementing a function to initialize the weights of the neural network
%  (randInitializeWeights.m)
%fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


%% =============== Part 7: Implement Backpropagation ===============
%  Once your cost matches up with ours, you should proceed to implement the
%  backpropagation algorithm for the neural network. You should add to the
%  code you've written in nnCostFunction.m to return the partial
%  derivatives of the parameters.
%
fprintf('\nChecking Backpropagation... \n');

%  Check gradients by running checkNNGradients
checkNNGradients;

%fprintf('\nProgram paused. Press enter to continue.\n');
%pause;


%% =============== Part 8: Implement Regularization ===============
%  Once your backpropagation implementation is correct, you should now
%  continue to implement the regularization with the cost and gradient.
%fprintf('\nChecking Backpropagation (w/ Regularization) ... \n')
%  Check gradients by running checkNNGradients
%lambda = 3;
%checkNNGradients(lambda);
% Also output the costFunction debugging values
%debug_J  = nnCostFunction(nn_params, input_layer_size, ...
 %                         hidden_layer_size, num_labels, X, y, lambda);
%fprintf(['\n\nCost at (fixed) debugging parameters (w/ lambda = %f): %f ' ...
 %        '\n(for lambda = 3, this value should be about 0.576051)\n\n'], lambda, debug_J);
%fprintf('Program paused. Press enter to continue.\n');
%pause;
%% =================== Part 8: Training NN ===================
%  You have now implemented all the code necessary to train a neural 
%  network. To train your neural network, we will now use "fmincg", which
%  is a function which works similarly to "fminunc". Recall that these
%  advanced optimizers are able to train our cost functions efficiently as
%  long as we provide them with the gradient computations.
%
fprintf('\nTraining Neural Network... \n')

%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', 50);

%  You should also try different values of lambda
lambda = 1;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

%fprintf('Program paused. Press enter to continue.\n');
%pause;


pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);


