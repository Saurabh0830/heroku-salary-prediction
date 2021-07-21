import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression

dataset=pd.read_csv("hiring.csv")

dataset.experience.fillna(0,inplace=True)
dataset.test_score.fillna(dataset.test_score.mean(),inplace=True)
# print(dataset)
X=dataset.iloc[:,:3]
Y=dataset.iloc[:,-1]

def word_to_number(word):
    d={0:0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10,"eleven":11,"twelve":12}
    return d[word]

X['experience']=X.experience.apply(lambda x:word_to_number(x))

model=LinearRegression()
model.fit(X,Y)

joblib.dump(model,"predict.pkl")

print("Model training is done")