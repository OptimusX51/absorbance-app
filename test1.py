import numpy as np
#import matplotlib.pyplot as plt  # To visualize
import pandas as pd  # To read data
from sklearn.linear_model import LinearRegression
import joblib as joblib
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle
#read new flavor data
test = pd.read_csv('PepsiNewFlavor2.csv')
# Load from file
with open('joblib_model.pkl', 'rb') as f:
    joblib_model = pickle.load(f)
# Calculate the accuracy and predictions
Ypredict = joblib_model.predict(test)
print(Ypredict)