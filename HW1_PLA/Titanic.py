import numpy as np
import pandas as pd
from PLA import PocketPLA , evaluate
from sklearn.linear_model import Perceptron

def preprocess(data):
    '''
    Data preprocess and selection
    '''
    # female -> 0 ; male -> 1 
    data['Sex'] = data['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    data['Age'] = data['Age'].fillna(data["Age"].mean())
    data['Fare'] = data['Fare'].fillna(data["Age"].mean())
    # data['Embarked'] = data['Embarked'].fillna('S')
    Y = data['Survived'].values
    data = data.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId','Survived','Embarked'],axis=1)
    X = data.values
    
    return X , Y 


    
if __name__ == "__main__":
    folder_root = "~/.kaggle/competitions/titanic/"
    train = pd.read_csv(folder_root+'train.csv',header=0)
    X , Y = preprocess(train)
    weight = PocketPLA(X,Y,50)
    pla = Perceptron(n_iter=50)
    pla = pla.fit(X,Y)
    print(pla.score(X,Y))

    test = pd.read_csv(folder_root+'test.csv',header=0)
    X , Y = preprocess(test)
    right , wrong = evaluate(X,Y,weight)
    print("Test dataset Acc : {}".format(right/(right+wrong)))
    print(pla.score(X,Y))
