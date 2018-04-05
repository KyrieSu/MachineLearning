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
        # print("iter {} : {}/{} = {}".format(iteration,right,total_sample,right/total_sample))
        iteration+=1
        if error_data==0:
            break
    return weight

def PocketPLA(X,Y,max_iter):
    total_sample , dim = X.shape
    weight = np.random.random_sample(dim)
    _ , least_error = evaluate(X,Y,weight)
    least_error_weight = weight

    for i in range(max_iter):
        index = random.randint(0,total_sample-1)
        x , y = X[index] , Y[index]
        pred = np.sign(np.dot(weight,x))
        label = np.sign(y)

        if pred!=label:
            tmp_weight = least_error_weight + (x*y)
            _ , tmp_error = evaluate(X,Y,tmp_weight)

            if tmp_error <= least_error:
                least_error = tmp_error
                least_error_weight = tmp_weight

        if least_error==0:
            break
    # print("Train dataset Acc : {} ".format((total_sample-least_error)/total_sample))
    # print(least_error_weight)
    return least_error_weight

def evaluate(testX,testY,weight):
    right , wrong = 0 , 0
    test_set = zip(testX,testY)
    for x , y in test_set:
        pred = np.sign(np.dot(weight,x))
        label = np.sign(y)
        if pred == label:
            right +=1
        else:
            wrong +=1
    total = right + wrong 
    # print("Accuracy : {} / {} = {}".format(right,total,right/total))
    # print(weight)
    return right , wrong


        

if __name__ == "__main__":
    # This is "linear separable" dataset
    datas = np.genfromtxt('train18.txt',dtype='float')
    X = datas[:,:-1]
    X = np.pad(X,((0,0),(1,0)),mode='constant', constant_values=1) # padding bias
    Y = datas[:,-1].astype(int)
    # NaivePLA(X,Y)
    weight = PocketPLA(X,Y,500)
    right , wrong = evaluate(X,Y,weight)
    print("Train dataset Acc : {}".format(right/(right+wrong)))
    '''
    Load Test dataset
    '''
    datas = np.genfromtxt('test18.txt',dtype='float')
    X = datas[:,:-1]
    X = np.pad(X,((0,0),(1,0)),mode='constant', constant_values=1) # padding bias
    Y = datas[:,-1].astype(int)
    right , wrong = evaluate(X,Y,weight)
    print("Test dataset Acc : {}".format(right/(right+wrong)))

