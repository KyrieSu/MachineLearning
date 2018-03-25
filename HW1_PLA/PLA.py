import numpy as np

def NaivePLA(row,dim,data):
    total_sample = row
    weight = np.random.random_sample(dim)
    iteration = 0
    while True:
        error_data = 0
        for x , y in data:
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

        

if __name__ == "__main__":
    # This is "linear separable" dataset
    data = np.genfromtxt('test_data.txt',dtype='float')
    X = data[:,:-1]
    Y = data[:,-1].astype(int)
    row , col = X.shape
    
    data = zip(X,Y)
    NaivePLA(row,col,data)

