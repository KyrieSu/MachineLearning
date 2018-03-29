import numpy as np
import random

def NaivePLA(X,Y):
    total_sample , dim = X.shape
    weight = np.random.random_sample(dim)
    iteration = 0
    datas = zip(X,Y)
    while True:
        error_data = 0
        for x , y in datas:
            pred = np.sign(np.dot(weight,x))
            label = np.sign(y)
            if pred!=label:
                error_data+=1
                weight+=(y*x)
        right = total_sample-error_data
        print("iter {} : {}/{} = {}".format(iteration,right,total_sample,right/total_sample))
        iteration+=1
        if error_data==0:
            break
    print(weight)

def PocketPLA(X,Y,max_iter):
    total_sample , dim = X.shape
    weight = np.random.random_sample(dim)
    datas = zip(X,Y)
    ##########
    def _cal_error_data(w):
        result = 0
        for x , y in datas:
            pred = np.sign(np.dot(weight,x))
            label = np.sign(y)
            if pred!=label:
                result+=1
        return result
    ##########
    least_error = _cal_error_data(weight)
    least_error_weight = weight

    for i in range(max_iter):
        index = random.randint(0,total_sample-1)
        x , y = X[index] , Y[index]
        pred = np.sign(np.dot(weight,x))
        label = np.sign(y)
        if pred!=label:
            tmp_weight = least_error_weight + (x*y)
            tmp_error = _cal_error_data(tmp_weight)

            if tmp_error <= least_error:
                least_error = tmp_error
                least_error_weight = tmp_weight
        right = total_sample - least_error
        print("iter {} : {}/{} = {}".format(i,right,total_sample,right/total_sample))
        print("Weight : {}".format(least_error_weight))
        if least_error==0:
            break
    print(least_error)
    print(least_error_weight)
    # return least_error_weight,least_error

        

if __name__ == "__main__":
    # This is "linear separable" dataset
    datas = np.genfromtxt('test18.txt',dtype='float')
    X = datas[:,:-1]
    Y = datas[:,-1].astype(int)
    row , col = X.shape
    NaivePLA(X,Y)
    # PocketPLA(X,Y,1000)

