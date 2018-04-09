import numpy as np
import random

def NaivePLA(X,Y):
    total_sample , dim = X.shape
    weight = np.random.rand(dim)

    while True:
        error_data = 0
        for x , y in zip(X,Y):
            pred = np.sign(np.dot(weight,x))
            label = np.sign(y)

            if pred!=label:
                error_data+=1
                weight+=(y*x)
        
        if error_data==0:
            break

    return weight

def PocketPLA(X,Y,max_iter):
    total_sample , dim = X.shape
    weight = np.random.rand(dim)
    _ , least_error = evaluate(X,Y,weight)
    least_error_weight = weight

    for i in range(max_iter):
        index = random.randint(0,total_sample-1)
        x , y = X[index] , Y[index]

        pred = np.sign(np.dot(weight,x))
        label = np.sign(y)
        if pred!=label:
            tmp_weight = weight + (x*y)
            _ , tmp_error = evaluate(X,Y,tmp_weight)

            weight = tmp_weight

            if tmp_error <= least_error:
                least_error = tmp_error
                least_error_weight = tmp_weight

        if least_error==0: # useless 
            break
    right = total_sample-least_error
    print("Accuracy : {} / {} = {}".format(right,total_sample,right/total_sample))
    
    return least_error_weight

def predict(testX,weight):
    '''
    predict the label of testX 
    return : the list of predict array
    '''
    preds = []
    for x in testX:
        tmp = np.sign(np.dot(weight,x))
        preds.append(tmp)
    assert(len(preds)==testX.shape[0])

    return np.asarray(preds).astype(int)

def evaluate(testX,testY,weight):
    right , wrong = 0 , 0
    preds = predict(testX,weight)
    for pred , label in zip(preds,testY):
        label = np.sign(label)
        if pred == label:
            right +=1
        else:
            wrong +=1
    total_sample = right + wrong 
    # print("Accuracy : {} / {} = {}".format(right,total_sample,right/total_sample))
    
    return right,wrong

        

# if __name__ == "__main__":
#     # all dataset is download by ML of Hsuan-Tien Lin
#     datas = np.genfromtxt('train18.txt',dtype='float')
#     X = datas[:,:-1]
#     Y = datas[:,-1].astype(int)
#     # weight = NaivePLA(X,Y)
#     weight = PocketPLA(X,Y,1000)
#     '''
#     Load Test dataset
#     '''
#     datas = np.genfromtxt('test18.txt',dtype='float')
#     X = datas[:,:-1]
#     Y = datas[:,-1].astype(int)
#     right , wrong = evaluate(X,Y,weight)
#     print("Test dataset Acc : {}".format(right/(right+wrong)))

