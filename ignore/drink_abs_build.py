import streamlit as st
import pandas as pd 
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pickle

st.write('''
# Softdrink Absorbance Prediction App

Absorption Prediction App  
       
''')

st.sidebar.header('User Input Paramenters')

def user_input_features():
    wavelength = st.sidebar.slider('Wavelength', 190.0, 500.0, 254.0, step=0.5)
    dilution_factor = st.sidebar.radio('Dilution Factor', [5000, 2000, 1000, 500, 100, 10, 5, 2], index=0)
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
          
    data = {'wavelength': wavelength,
            'dilution_factor': dilution_factor,
            'Water':0, 'Carbonated.water':0, 'HFCS':0,
            'Concentrated.OJ':0, 'Caramel.color':0, 'Sugar':0, 'Sucralose':0, 'Acesulfame.potassium':0,
            'Aspartame':0, 'Dextrose':0, 'Phosphoric.acid':0, 'Citric.acid':0, 'Caffeine':0, 'Salt':0,
            'Sodium.citrate':0, 'Potassium.citrate':0, 'Sodium.benzoate':0, 'Potassium.benzoate':0,
            'Sodium.polyphosphates':0, 'Sodium.hexametaphosphate':0, 'Monopotassium.phopshate':0, 'Potassium.sorbate':0,
            'Erythorbic.acid':0, 'Modified.corn.starch':0, 'Gum.arabic':0, 'Glycerol.ester.of.rosin':0,
            'Calcium.disodium.EDTA':0, 'Citrus.flavor':0, 'Fruit.punch':0, 'Grape':0, 'Cola':0, 'Cherry.Cola':0,
            'Moutain.Dew':0, 'Root.Beer':0, 'Red40':0, 'Yellow5':0, 'Blue1':0, 'Panax.Ginseng':0, 'Citrus.Pectin':0,
            'Ascorbic.acid':0, 'Calcium.pentothenate':0, 'Niacinamide':0, 'Vitamine.E.acetate':0,
            'Pyridoxine.hydrochloride':0, 'Quillaia.extract':0
            }
    
    #data = {'f0': wavelength,
    #        'f1': dilution_factor,
    #        'f2':0, 'f3':0, 'f4':0,'f5':0, 'f6':0, 'f7':0, 'f8':0, 'f9':0,
    #        'f10':0, 'f11':0, 'f12':0, 'f13':0, 'f14':0, 'f15':0,
    #        'f16':0, 'f17':0, 'f18':0, 'f19':0,
    #        'f20':0, 'f21':0, 'f22':0, 'f23':0,
    #        'f24':0, 'f25':0, 'f26':0, 'f27':0,
    #        'f28':0, 'f29':0, 'f230':0, 'f31':0, 'f32':0, 'f33':0,
    #        'f34':0, 'f35':0, 'f36':0, 'f37':0, 'f38':0, 'f39':0, 'f40':0,
    #        'f41':0, 'f42':0, 'f43':0, 'f44':0,'f45':0, 'f46':0
    #        }

    for ingredient in ingredients:
        data[ingredient]=1
         

    st.sidebar.write(ingredients)
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()
df = df.dropna()

st.subheader('User Input Features')
st.write(df)

###create model pickle######################################################################################
datset = pd.read_csv('Pepsi Data Frame.csv').values

#X = datset.data
#Y = datset.target

X = datset[:,:-1]
Y = datset[:,len(datset[0])-1]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0) 

train = xgb.DMatrix(x_train, label=y_train)
test = xgb.DMatrix(x_test, label=y_test)


param = {
    'max_depth': 4,
    'eta': 0.3,
    'objective': 'multi:softmax',
    'num_class': 3
}
epochs = 10

#model = xgb.train(param, train, epochs)

xg_reg = xgb.XGBClassifier(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)
xg_reg.fit(x_train,y_train)

##Save Model
#xg_reg.save_model("model.json")

st.subheader('Class labels and their corresponding index number')
st.write(df.columns)

#names=['Water','Carbonated.water','HFCS','Concentrated.OJ','Caramel.color','Sugar','Sucralose','Acesulfame.potassium','Aspartame','Dextrose','Phosphoric.acid','Citric.acid','Caffeine','Salt','Sodium.citrate','Potassium.citrate','Sodium.benzoate','Potassium.benzoate','Sodium.polyphosphates','Sodium.hexametaphosphate','Monopotassium.phopshate','Potassium.sorbate','Erythorbic.acid','Modified.corn.starch','Gum.arabic','Glycerol.ester.of.rosin','Calcium.disodium.EDTA','Citrus.flavor','Fruit.punch','Grape','Cola','Cherry.Cola','Moutain.Dew','Root.Beer','Red40','Yellow5','Blue1','Panax.Ginseng','Citrus.Pectin','Ascorbic.acid','Calcium.pentothenate','Niacinamide','Vitamine.E.acetate','Pyridoxine.hydrochloride','Quillaia.extract']
#names = xg_reg.get_booster().feature_names
#names=xg_reg.get_booster().feature_names=df.columns
#names = xg_reg.get_booster()
#st.write(names)
#prediction1 = xg_reg.predict(df[names].iloc[[-1]]).all()
test1=[[Wavelength,Dilution,Water,Carbonated.water,HFCS,Concentrated.OJ,Caramel.color,Sugar,Sucralose,Acesulfame.potassium,Aspartame,Dextrose,Phosphoric.acid,Citric.acid,Caffeine,Salt,Sodium.citrate,Potassium.citrate,Sodium.benzoate,Potassium.benzoate,Sodium.polyphosphates,Sodium.hexametaphosphate,Monopotassium.phopshate,Potassium.sorbate,Erythorbic.acid,Modified.corn.starch,Gum.arabic,Glycerol.ester.of.rosin,Calcium.disodium.EDTA,Citrus.flavor,Fruit.punch,Grape,Cola,Cherry.Cola,Moutain.Dew,Root.Beer,Red40,Yellow5,Blue1,Panax.Ginseng,Citrus.Pectin,Ascorbic.acid,Calcium.pentothenate,Niacinamide,Vitamine.E.acetate,Pyridoxine.hydrochloride],
[265,5,1,0,0,0,0,0,1,1,0,0,0,1,0,1,1,0,0,0,0,0,1,0,0,0,1,1,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]]


prediction = xg_reg.predict(test1)
#prediction = xg_reg.predict(df.values)
#######################################################################################################

#st.subheader('Class labels and their corresponding index number')
#st.write(iris.target_names)

st.subheader('Prediction')
st.write(x_test[:1,:]) #all columns, first row

st.write(df.values)
st.write(prediction)
#st.write(prediction)

#st.subheader('Prediction Probability')
#st.write(prediction_proba)

