# Geoffrey Hughes
# Erik Linstead
# 002306123
# ghughes@chapman.edu
# CPSC 392-01
# Assignment 3: Logistic Regression



#logistic regression
import math
import random

TRAIN_DATA_FILE = "temps_fever.csv"

#read the train file and return the data matrix and the target variable to predict
def readData(fname):
    data = []
    labels = []
    f = open(fname,"r")
    for i in f:
        instance = i.split(",")
        data.append(float(instance[0]))
        labels.append(float(instance[1]))
        
        print(instance[0])
        print(instance[1])
        
    f.close()
    return [data,labels]

def logistic_fn(b0,b1,x):
    output = b0+b1*x
    return 1/(1+math.exp(-1*(output)))

def logistic_fn_prime(b0,b1,x):
    pred = logistic_fn(b0,b1,x)
    return pred*(1-pred)

def train_epoch(data,labels,b0,b1,rate):
    for i in range(len(data)):
        estimate = logistic_fn(b0,b1,data[i])
        residual = labels[i]-estimate
        newb0 = b0 + rate*residual*logistic_fn_prime(b0,b1,data[i])
        newb1 = b1 + rate*residual*logistic_fn_prime(b0,b1,data[i])*data[i]
        b0=newb0
        b1=newb1
    return [b0,b1]

def train(data,labels,b0,b1,rate,epochs):
    count = 0
    param_history = []
    param_history.append([b0,b1])
    while count < epochs:
        adjs = train_epoch(data,labels,b0,b1,rate)
        param_history.append(adjs)
        b0 = adjs[0]
        b1 = adjs[1]
        count += 1
    return [b0,b1,param_history]

def make_pred(b0,b1,input_v):
    result = 0
    raw = logistic_fn(b0,b1,input_v)
    if raw >= .5:
        result = 1
    return result

if __name__ == "__main__":
	train_matrix = readData(TRAIN_DATA_FILE)
	b0 = (random.random() / 10)
	b1 = (random.random() / 10)
	learning_rate = .01
	epoch_limit = 5000
    
	model = train(train_matrix[0],train_matrix[1],b0,b1,learning_rate,epoch_limit)
	print("The b0 param is: ",model[0])
	print("The b1 param is: ",model[1])
    
        #print(make_pred(model[0], model[1], 98.6))
        #print(make_pred(model[0], model[1], 111.0))
    
        
        