import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.metrics import accuracy_score , precision_score, recall_score, confusion_matrix , f1_score, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

train = pd.read_csv("C:\\Users\\Hari\\Google Drive\\Trails Hackathons\\Miskaa\\train.csv")
test = pd.read_csv("C:\\Users\\Hari\\Google Drive\\Trails Hackathons\\Miskaa\\test.csv")


train = train.drop(columns= ["PassengerId","Name","Ticket","Cabin"],axis=1)
test = test.drop(columns= ["PassengerId","Name","Ticket","Cabin"],axis=1)

train["Age"] , train["Embarked"] = train["Age"].fillna(train["Age"].median()),train["Embarked"].fillna(train["Embarked"].mode()[0])
test["Age"], test["Embarked"] = test["Age"].fillna(test["Age"].median()),test["Embarked"].fillna(test["Embarked"].mode()[0])


le = LabelEncoder()
train["Embarked"] = le.fit_transform(train["Embarked"])
train["Sex"] = le.fit_transform(train["Sex"])
test["Embarked"] = le.fit_transform(test["Embarked"])
test["Sex"] = le.fit_transform(test["Sex"])

train["Sex"], train["Embarked"]= pd.get_dummies(train["Sex"],drop_first=True), pd.get_dummies(train["Embarked"],drop_first =True)
test["Sex"], test["Embarked"]= pd.get_dummies(test["Sex"],drop_first=True), pd.get_dummies(test["Embarked"],drop_first= True)

def normalization(i):
    x = (i-i.min())/(i.max()-i.min())
    return(x)   

train = normalization(train)
test = normalization(test)

x = train.drop(columns= ["Survived"],axis=1)
y = train["Survived"]

xtrain , xvalid , ytrain , yvalid = train_test_split(x,y,test_size= 0.3 , random_state = 45)

from lazypredict.Supervised import LazyClassifier
clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(xtrain, xvalid, ytrain, yvalid)
print(models)
