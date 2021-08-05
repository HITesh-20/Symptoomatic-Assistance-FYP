import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
import warnings
warnings.filterwarnings("ignore")


#READ DATA
df1=pd.read_csv("Heart Disease1.csv")
df1=df1.drop('no',axis='columns')
x=df1.drop('TenYearCHD',axis='columns')
y=df1.TenYearCHD
y = y.astype('int')
x = x.astype('int')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
#FITTING MODEL
lr=LogisticRegression().fit(x_train,y_train)

pickle.dump(lr, open("pickle.pkl", "wb"))
