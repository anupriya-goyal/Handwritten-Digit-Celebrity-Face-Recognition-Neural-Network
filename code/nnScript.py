


import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import random
import time
import pickle


def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer
    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W
def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    return  1.0 / (1.0 + np.exp(-z))
def preprocess():
    mat = loadmat('mnist_all.mat')
    
    pre_train_data = np.zeros(shape=(50000, 784))
    pre_valid_data = np.zeros(shape=(10000, 784))
    pre_test_data = np.zeros(shape=(10000, 784))
    pre_tr_lb = np.zeros(shape=(50000,))
    pre_vd_lb = np.zeros(shape=(10000,))
    pre_ts_lb = np.zeros(shape=(10000,))
    
    init_tr_len = 0
    init_vd_len = 0
    init_ts_len = 0
    init_lb_len = 0
    init_vd_lb_len = 0
    #splitting data into 6 arrays
    #for trainging set, test set and validation set
    #recored labels accordingly
    for id in mat:
        
        if "train" in id:
            lb = id[-1]  
            tp = mat.get(id)
            tp_sh = range(tp.shape[0])
            per_tp = np.random.permutation(tp_sh)
            len_tp = len(tp) 
            len_lb =len_tp  - 1000 

           
            pre_train_data[init_tr_len:init_tr_len + len_lb] = tp[per_tp[1000:], :]
            init_tr_len += len_lb

            pre_tr_lb[init_lb_len:init_lb_len + len_lb] = lb
            init_lb_len += len_lb

           
            pre_valid_data[init_vd_len:init_vd_len + 1000] = tp[per_tp[0:1000], :]
            init_vd_len += 1000

            pre_vd_lb[init_vd_lb_len:init_vd_lb_len + 1000] = lb
            init_vd_lb_len += 1000

           
        elif "test" in id:
            lb = id[-1]
            tp = mat.get(id)
            tp_sh = range(tp.shape[0])
            per_tp = np.random.permutation(tp_sh)
            len_tp = len(tp)
            pre_ts_lb[ init_ts_len:init_ts_len + len_tp ] = lb
            pre_test_data[init_ts_len:init_ts_len + len_tp ] = tp[per_tp]
            init_ts_len += len_tp
           
    size_of_tr = range(pre_train_data.shape[0])
    per_tr = np.random.permutation(size_of_tr)
    train_data = pre_train_data[per_tr]
    train_data = np.double(train_data)
    train_data = train_data / 255.0
    train_label = pre_tr_lb[per_tr]

    size_of_val = range(pre_valid_data.shape[0])
    per_val = np.random.permutation(size_of_val)
    validation_data = pre_valid_data[per_val]
    validation_data = np.double(validation_data)
    validation_data = validation_data / 255.0
    validation_label = pre_vd_lb[per_val]

    size_of_ts = range(pre_test_data.shape[0])
    per_ts = np.random.permutation(size_of_ts)
    test_data = pre_test_data[per_ts]
    test_data = np.double(test_data)
    test_data = test_data / 255.0
    test_label = pre_ts_lb[per_ts]
    

    # Feature selection
    # Your code here.
   
   
    
    
    div_train=np.std(train_data,0)
   
  
    min_div_train= min(div_train)
   
    
    
    included_features=[]
    removed_features=[]
    
    for i in range(train_data.shape[1]):
       
        if (div_train[i]>0.002):
            included_features.append(i)
        else:
            removed_features.append(i)
            
   
    train_data = np.delete(train_data, removed_features,1)
    validation_data = np.delete(validation_data,removed_features,1)
    test_data = np.delete(test_data, removed_features,1)
   
    print('preprocess done')
    
    return train_data, train_label, validation_data, validation_label, test_data, test_label,included_features

def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
#     %   likelihood error function with regularization) given the parameters 
#     %   of Neural Networks, thetraining data, their corresponding training 
#     %   labels and lambda - regularization hyper-parameter.
#     % Input:
#     % params: vector of weights of 2 matrices w1 (weights of connections from
#     %     input layer to hidden layer) and w2 (weights of connections from
#     %     hidden layer to output layer) where all of the weights are contained
#     %     in a single vector.
#     % n_input: number of node in input layer (not include the bias node)
#     % n_hidden: number of node in hidden layer (not include the bias node)
#     % n_class: number of node in output layer (number of classes in
#     %     classification problem
#     % training_data: matrix of training data. Each row of this matrix
#     %     represents the feature vector of a particular image
#     % training_label: the vector of truth label of training images. Each entry
#     %     in the vector represents the truth label of its corresponding image.
#     % lambda: regularization hyper-parameter. This value is used for fixing the
#     %     overfitting problem.
       
#     % Output: 
#     % obj_val: a scalar value representing value of error function
#     % obj_grad: a SINGLE vector of gradient value of error function
#     % NOTE: how to compute obj_grad
#     % Use backpropagation algorithm to compute the gradient of error function
#     % for each weights in weight matrices.
#     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#     % reshape 'params' vector into 2 matrices of weight w1 and w2
#     % w1: matrix of weights of connections from input layer to hidden layers.
#     %     w1(i, j) represents the weight of connection from unit j in input 
#     %     layer to unit i in hidden layer.
#     % w2: matrix of weights of connections from hidden layer to output layers.
#     %     w2(i, j) represents the weight of connection from unit j in hidden 
#     %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0
    obj_grad = np.array([])
    
    # Your code here
    train_data_size=(training_data.shape[0])
    bias_term_1 = np.ones((train_data_size, 1))

    input_data_X = np.concatenate((training_data,bias_term_1), axis = 1)

    
    res_1=np.dot(w1,input_data_X.T)
    
    z=sigmoid(res_1)
    z=z.T
    z=np.array(z)
    
    
    bias_term_2 = np.ones((len(z), 1))

    z = np.concatenate((z,bias_term_2), axis = 1)
    

    res_2=np.dot(w2,z.T)
    
    output=sigmoid(res_2)
   
   
    gen_lab=np.zeros((training_label.shape[0],n_class),dtype=np.int)
    for i in range(training_label.shape[0]):
        for index in range(n_class):
            if(index==int(training_label[i])):
                  gen_lab[i][index]=1
    Y=gen_lab
    Y=Y.T
   
    E_Funct=(1 - Y)*np.log(1 - output) + (Y*np.log(output))  
    E_Funct1=-(1/train_data_size)*(np.sum(E_Funct[:,:]))
    delta_op=output-Y
    error_w2=np.dot(delta_op,z)
    
    delta_hidden=np.dot(w2.T,delta_op)*(z.T)*(1-z.T)

    error_w1=np.dot(delta_hidden,input_data_X)
    error_w1=error_w1[:-1,:]
    R_term=(lambdaval/(2*train_data_size))*(np.sum(w1**2)+np.sum(w2**2))
    obj_val=E_Funct1+R_term
  
     
    grad_w1=(lambdaval*w1+error_w1)/train_data_size
    grad_w2=(lambdaval*w2+error_w2)/train_data_size
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
  
    
    
    
    
    
    

    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    

    return (obj_val, obj_grad)
  
def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
#     % Network.
#     % Input:
#     % w1: matrix of weights of connections from input layer to hidden layers.
#     %     w1(i, j) represents the weight of connection from unit i in input 
#     %     layer to unit j in hidden layer.
#     % w2: matrix of weights of connections from hidden layer to output layers.
#     %     w2(i, j) represents the weight of connection from unit i in input 
#     %     layer to unit j in hidden layer.
#     % data: matrix of data. Each row of this matrix represents the feature 
#     %       vector of a particular image
       
#     % Output: 
#     % label: a column vector of predicted labels"""

    labels = np.array([])
    # Your code here
    input_data=data.shape[0]
    bias_term = np.ones((input_data, 1))
    input_vector_x = np.concatenate((data,bias_term), axis = 1)
    
    output_label=np.dot(w1,input_vector_x.T)
    
    z=sigmoid(output_label)
    z=z.T
    bias_term_1 = np.ones((len(z), 1))

    z= np.concatenate((z,bias_term_1), axis = 1)

    final_output=np.dot(w2,z.T)
    
    Output=sigmoid(final_output)
    labels=(np.argmax(Output,0))
    return labels

train_data, train_label, validation_data, validation_label, test_data, test_label,features = preprocess()
#train_data1=np.array(train_data)
#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = (train_data.shape[1])

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50

# set the number of nodes in output unit
n_class = 10

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

# set the regularization hyper-parameter
lambdaval = 10

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter': 50}  # Preferred value.

ts=time.time()
nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
te=time.time()

# In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
# and nnObjGradient. Check documentation for this function before you proceed.
# nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


# Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

# Test the computed parameters

predicted_label = nnPredict(w1, w2, train_data)

# find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, validation_data)

# find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, test_data)

# find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

obj=[features,n_hidden,w1,w2,lambdaval]
with open('params.pickle','wb') as f:
    pickle.dump(obj,f)

print("Time required: ",te-ts)