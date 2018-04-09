import numpy as np
import pandas as pd
from PLA import PocketPLA , predict
from sklearn.linear_model import Perceptron
import csv as csv


def preprocess(data):
    '''
    Data preprocess and selection
    '''
    # female -> 0 ; male -> 1 
    data['Sex'] = data['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    
    data['Age'] = data['Age'].fillna(data["Age"].mean())
    data.loc[data['Age'] <= 16 , 'Age'] = 0
    data.loc[(data['Age'] > 16) & (data['Age'] <= 32), 'Age'] = 1
    data.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'Age'] = 2
    data.loc[data['Age'] > 48 , 'Age'] = 4

    data['Fare'] = data['Fare'].fillna(data["Fare"].mean())
    data.loc[data['Fare']<= 8 , 'Fare'] = 0
    data.loc[(data['Fare'] > 8) & (data['Fare'] <= 15), 'Fare'] = 1
    data.loc[(data['Fare'] > 15) & (data['Fare'] <= 31), 'Fare'] = 2
    data.loc[data['Fare'] > 31 , 'Fare'] = 4
    data = data.drop(['Name','Ticket', 'Cabin', 'PassengerId','Survived','Embarked'],axis=1)
    X = data.values

    return X


    
if __name__ == "__main__":
    folder_root = "~/.kaggle/competitions/titanic/"
    train = pd.read_csv(folder_root+'train.csv',header=0)
    Y = train['Survived'].values
    Y[Y==0] = -1
    X = preprocess(train)
    weight = PocketPLA(X,Y,1000)
    pla = Perceptron(n_iter=50)
    pla = pla.fit(X,Y)

    test = pd.read_csv(folder_root+'test.csv',header=0)
    ids = test['PassengerId'].values
    X = preprocess(test)
    sk_preds = pla.predict(X)
    preds = predict(X,weight)
    preds[preds==-1] = 0
    sk_preds[sk_preds==-1] = 0
    right , wrong = 0 , 0
    
    with open("myplaFinal1.csv", "w") as predictions_file:
        open_file_object = csv.writer(predictions_file)
        open_file_object.writerow(["PassengerId","Survived"])
        open_file_object.writerows(zip(ids, preds))
    predictions_file.close()
    print ('Done.')

