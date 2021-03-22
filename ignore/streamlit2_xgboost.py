import streamlit as st
import pandas as pd 
import numpy as np
#import pandas_ml as pdml
from sklearn import datasets
#from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score

st.write('''
# Simple Iris Flower Prediction App

This app predicts the Iris flower type!  
       
''')

st.sidebar.header('User Input Paramenters')

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal Length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal Width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal Length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal Width', 0.1, 2.5, 0.2)
    data = {'f0': sepal_length,
            'f1': sepal_width,
            'f2': petal_length,
            'f3': petal_width}
    
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()
x, y = df.iloc[:,:-4],df.iloc[:,-1]
df_dmatrix = xgb.DMatrix(data=x, label=y)

st.subheader('User Input Features')
st.write(df)

iris = datasets.load_iris()
X = iris.data
Y = iris.target

x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=0) 

train = xgb.DMatrix(x_train, label=y_train)
test = xgb.DMatrix(x_test, label=y_test)


param = {
    'max_depth': 4,
    'eta': 0.3,
    'objective': 'multi:softmax',
    'num_class': 3
}
epochs = 10

model = xgb.train(param, train, epochs)

xg_reg = xgb.XGBClassifier(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)
xg_reg.fit(x_train,y_train)

names = xg_reg.get_booster().feature_names
prediction1 = xg_reg.predict(df[names].iloc[[-1]])

prediction = xg_reg.predict(x_test)

st.subheader('Class labels and their corresponding index number')
st.write(iris.target_names)

st.subheader('Prediction')
st.write(iris.target_names[prediction1])