


import numpy as np
import pickle
import math
from scipy.optimize import minimize
from scipy.io import loadmat
import time

# Do not change this
def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
                            
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    epsilon = math.sqrt(6) / math.sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W

def classAssignFunction(training_label,n_class):
    #output label of size row * no of class
    output_label = np.zeros((training_label.shape[0],n_class),dtype=np.int)
    for row in range(training_label.shape[0]):
        for col in range(n_class):
            if(col==int(training_label[row])):
                output_label[row][col]=1
    return output_label

# Replace this with your sigmoid implementation
def sigmoid(z):
    denominator_value = (1.0 + np.exp(-z))
    return  1.0 / denominator_value

def errorFunction(Y,output,n):
    t1=Y*np.log(output)
    t2=(1 - Y)*np.log(1 - output) 
    out =t1 + t2 
    
    return -1*(np.sum(out[:,:])/n)

# Replace this with your nnPredict implementation
def nnPredict(w1,w2,data):
  
    length = (data.shape[0])
    b=np.ones((length,1))
    
    X=np.concatenate((data,b),axis=1)
    
    X_input = np.dot(w1,X.T)
    
    #output 
    hidden_out = sigmoid(X_input)
    
    hidden_output = hidden_out.T
    hidden_length = (hidden_output.shape[0])
    b1=np.ones((hidden_length,1))
    
    B=np.concatenate((hidden_output,b1),axis=1)
    output = np.dot(w2,B.T)
    
    #empty label array
    labels = np.array([])
    labels =np.argmax(sigmoid(output),0)
    
    return labels

# Replace this with your nnObjFunction implementation
def nnObjFunction(params, *args):
    
    n_input=args[0]
    n_hidden=args[1]
    n_class=args[2]
    training_data= args[3]
    training_label=args[4]
    lambdaval=args[5]
    #n_input, n_hidden, n_class, training_data, training_label, lambdaval = args
    #w1 weight between input layers and hidden layer
    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    #w2 weight between hidden layers and output layer
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    
    
    #number of input 
    length =(training_data.shape[0])
    b=np.ones((length,1))
    
    X = np.concatenate((training_data,b),axis=1) 
    
    #Input at hidden layer
    hidden_input = np.dot(w1,X.T)
    
    #output at hidden layer with Activation function
    hidden_out = sigmoid(hidden_input)
    
   
    #output at hidden layer after applying activation function
    hidden_out = hidden_out.T
    hidden_out = np.array(hidden_out)
    lenght2 = len(hidden_out)
    
    b1= np.ones((lenght2,1))
  
    hidden_out = np.concatenate((hidden_out,b1),axis=1)

    B= np.dot(w2,hidden_out.T)
    
    #output at output layer after applying activation function
    output=sigmoid(B)
    
    #Assign class based on output
    Y = classAssignFunction(training_label,n_class)
    Y=Y.T
    #error = errorFunction(Y,output,training_data.shape[0])
    #t1=Y*np.log(output)
    t1= Y*np.log(output)
    #t2=(1 - Y)*np.log(1 - output) 
    t2=(1-Y)*np.log(1 - output)
    out =t1 + t2 
    n = training_data.shape[0]
    error =-1*(np.sum(out[:,:])/n)
    
    
    #deviation in output
    output_difference = output -Y
    
    w2err = np.dot(output_difference,hidden_out)
    
    delta = hidden_out.T*(1-hidden_out.T)
    
    delta_hidden = np.dot(w2.T,output_difference)*delta
    w1err = np.dot(delta_hidden,X)
    w1err =w1err[:-1,:]
    
    #Regularization factor calculation
    Reg_factor=(lambdaval/(2*n))*(np.sum(w1**2)+np.sum(w2**2))
    obj_val=0
    obj_val=error+Reg_factor
    obj_grad = np.array([])
    
     
    #gradient descent value at w1  
    grad_w1=(lambdaval*w1+w1err)/n
    #gradient descent value at w2  
    grad_w2=(lambdaval*w2+w2err)/n
                                  
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
                                  
    return (obj_val, obj_grad)

# Do not change this
def preprocess():
    pickle_obj = pickle.load(file=open('face_all.pickle', 'rb'))
    features = pickle_obj['Features']
    labels = pickle_obj['Labels']
    train_x = features[0:21100] / 255
    valid_x = features[21100:23765] / 255
    test_x = features[23765:] / 255

    labels = labels[0]
    train_y = labels[0:21100]
    valid_y = labels[21100:23765]
    test_y = labels[23765:]
    return train_x, train_y, valid_x, valid_y, test_x, test_y

"""**************Neural Network Script Starts here********************************"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
#  Train Neural Network
# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]
# set the number of nodes in hidden unit (not including bias unit)
n_hidden =256 #default 256
# set the number of nodes in output unit
n_class = 2  

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);
# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)
# set the regularization hyper-parameter
lambdaval = 25 #default 0
args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# print('args')
# print(args[0])
# print(args[1])
# print(args[2])
# print(args[3])
# print(args[4])
# print(args[5])
#print(args(1))
#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
opts = {'maxiter' :50}    # Preferred value.
ts= time.time()
nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)
params = nn_params.get('x')
#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

#Test the computed parameters
predicted_label = nnPredict(w1,w2,train_data)
#find the accuracy on Training Dataset
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,validation_data)
#find the accuracy on Validation Dataset
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,test_data)
#find the accuracy on Validation Dataset
print('\n Test set Accuracy:' +  str(100*np.mean((predicted_label == test_label).astype(float))) + '%')
te= time.time()

print("time require to run :\n",te-ts)