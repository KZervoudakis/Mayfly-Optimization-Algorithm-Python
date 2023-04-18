import numpy as np

def ContinousCrossover(x1,x2,gamma):
    #print(x1)
    #print(x2)
    alpha = np.random.uniform(-gamma,1+gamma,size=(x1.shape[0], x1.shape[1]))
    y1=alpha*x1+(1-alpha)*x2;
    y2=alpha*x2+(1-alpha)*x1; 
    #print(y1)
    #print(y2)
    return y1, y2

def ContinousMutation(x1,problem):
    #print(x1)
    #print(x2)
    y1=x1+0.1*(problem['VarMax']-problem['VarMin'])*np.random.uniform(low=-1, high=1, size=(x1.shape[0], x1.shape[1]))
    #print(y1)               np.random.uniform(low=-1, high=1, size=(x1.shape[0], x1.shape[1]))
    #print(y2)
    return y1