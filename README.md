This repository contains python codes of machine learning modules

# FOLDER: NeuralNetwork_implementation/ #

## neural_network.py ##

### class "classification" ###

Feedforward artificial network. Using gradient descent, forward, backward propagation to optimize weights.  
Weights between neurons are initialized randomly in the range of [-4/sqrt(d), 4/sqrt(d))  
See **example_neural_network.py** for an example of using the code

#### Input Parameters ####
* nf: number of features
* hidden_sl: a tuple with # of neurons of layers
* lrate: learning rate (default: 60.)
* penalty: l2 regularization penalty parameter (default: 0.0001)
* maxiter: maximum # of iterations (default: 50000)
* tol: tolerance value of cost function (default: 1e-6)

#### Attributes ####
* fit(x, y): do the neural network fitting
* predict(): input a 1-D array of x, output predicted y value with probability

## Example codes ##

### example_neural_network.py ###
An example code for using the neural_network.classfication() class
by inputing a set of XOR data:  
0 0 False  
0 1 True  
1 0 True  
1 1 False  
The fitting results (probability map) can be seen in  
**fittiingResult_exNeuralNetwork1_1.png** and **fittingResult_exNeuralNetwork1_2.png**

### example_logistic.py ###
An example code for using **scikit-learn** to do logistic fitting, with input of
data.xyz (my simulation data)  
The fitting results can be seen in **fitting_result_exLogistic.png**,
where (yellow, purple) represent (True, False), and 3 ovals are 0.2, 0.5, 0.8
from the fitting results of logistic regression

# FOLDER: EmailClassification_sklearn_naiveBayes/ #
This folder contains an email classification code:    
Using naive Bayes method to determine if an email belongs to author 1 or author 2
