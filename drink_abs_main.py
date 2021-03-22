from xgboost.core import Booster
import streamlit as st
import pandas as pd 
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pickle
import joblib

st.write('''
# Softdrink Absorbance Prediction App

Absorbance Prediction App  
       
''')

st.sidebar.header('User Input Paramenters')

def user_input_features():
    Wavelength = st.sidebar.slider('Wavelength', 190.0, 500.0, 254.0, step=0.5)
    Dilution = st.sidebar.radio('Dilution Factor', [5000, 2000, 1000, 500, 100, 10, 5, 2], index=0)
    ingredients_list =[
        'Water', 'Carbonated.water', 'HFCS',
        'Concentrated.OJ', 'Caramel.color', 'Sugar', 'Sucralose', 'Acesulfame.potassium',
        'Aspartame', 'Dextrose', 'Phosphoric.acid', 'Citric.acid', 'Caffeine', 'Salt',
        'Sodium.citrate', 'Potassium.citrate', 'Sodium.benzoate', 'Potassium.benzoate',
        'Sodium.polyphosphates', 'Sodium.hexametaphosphate', 'Monopotassium.phopshate', 'Potassium.sorbate',
        'Erythorbic.acid', 'Modified.corn.starch', 'Gum.arabic', 'Glycerol.ester.of.rosin',
        'Calcium.disodium.EDTA', 'Citrus.flavor', 'Fruit.punch', 'Grape', 'Cola', 'Cherry.Cola',
        'Moutain.Dew', 'Root.Beer', 'Red40', 'Yellow5', 'Blue1', 'Panax.Ginseng', 'Citrus.Pectin',
        'Ascorbic.acid', 'Calcium.pentothenate', 'Niacinamide', 'Vitamine.E.acetate',
        'Pyridoxine.hydrochloride', 'Quillaia.extract'
    ]
    ingredients = st.sidebar.multiselect('Ingredients',ingredients_list, default='Water')
          
    data = {'Wavelength': Wavelength,
            'Dilution': Dilution,
            'Water':0, 'Carbonated.water':0, 'HFCS':0,
            'Concentrated.OJ':0, 'Caramel.color':0, 'Sugar':0, 'Sucralose':0, 'Acesulfame.potassium':0,
            'Aspartame':0, 'Dextrose':0, 'Phosphoric.acid':0, 'Citric.acid':0, 'Caffeine':0, 'Salt':0,
            'Sodium.citrate':0, 'Potassium.citrate':0, 'Sodium.benzoate':0, 'Potassium.benzoate':0,
            'Sodium.polyphosphates':0, 'Sodium.hexametaphosphate':0, 'Monopotassium.phopshate':0, 'Potassium.sorbate':0,
            'Erythorbic.acid':0, 'Modified.corn.starch':0, 'Gum.arabic':0, 'Glycerol.ester.of.rosin':0,
            'Calcium.disodium.EDTA':0, 'Citrus.flavor':0, 'Fruit.punch':0, 'Grape':0, 'Cola':0, 'Cherry.Cola':0,
            'Moutain.Dew':0, 'Root.Beer':0, 'Red40':0, 'Yellow5':0, 'Blue1':0, 'Panax.Ginseng':0, 'Citrus.Pectin':0,
            'Ascorbic.acid':0, 'Calcium.pentothenate':0, 'Niacinamide':0, 'Vitamine.E.acetate':0,
            'Pyridoxine.hydrochloride':0
            }

    for ingredient in ingredients:
        data[ingredient]=1
        
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input Features')
st.write(df)

###load model file ###############
test = pd.read_csv('PepsiNewFlavor2.csv')
with open('joblib_model.pkl', 'rb') as f:
    joblib_model = pickle.load(f)

#  Calculate the predictions
Ypredict = joblib_model.predict(df)
###################################

st.subheader('Absorbance Prediction')
st.write(Ypredict)