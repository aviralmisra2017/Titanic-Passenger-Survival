import streamlit as st
import pandas as pd
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
from tensorflow.keras.models import load_model
import pickle
import os

st.title("Passenger survival chance in Titanic Journey")

pclass=st.slider("Enter passenger class for the user",1,3)
sex=st.selectbox('Enter the passenger Gender',['male','female'])
sibsp=st.slider('Enter the Passengers Sibling Spouse',1,8)
parch=st.slider('Enter the Passengers total parents and child',0,6)
fare=st.number_input('Enter the Fare',0,100)
embarked=st.selectbox('Enter passenger station from where they started the journey',['Southampton','Chebourg','Queenstown'])

data=pd.DataFrame([{'Pclass':pclass,'Sex':sex,'SibSp':sibsp,'Parch':parch,'Fare':fare,'Embarked':embarked}])

model = load_model('model.keras')
with open('label_encoder.pkl', 'rb') as f:
    label=pickle.load(f)

with open('onehot_encoder.pkl', 'rb') as f:
    onehot=pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler=pickle.load(f)

data['Sex']=label.transform(data['Sex'])
embarked=onehot.transform(data[['Embarked']])
embarked=pd.DataFrame(embarked,columns=onehot.get_feature_names_out())

data=pd.concat([data.drop(columns=['Embarked']),embarked],axis=1)
data[['Pclass','SibSp','Parch','Fare']]=scaler.transform(data[['Pclass','SibSp','Parch','Fare']])

X = data.to_numpy(dtype='float32')
y = model(X, training=False).numpy()[0][0]
def Chance(y):
    if y>0.5:
        return 'Passenger will survive journey'
    else:
        return 'Passenger will not survive journey'

if st.button('Predict Survival Chance'):
    st.write("Probability of Passenger Survival chance: ",y)
    st.write(Chance(y))