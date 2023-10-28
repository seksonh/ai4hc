import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Estimation of obesity levels based on eating habits and physical condition

This app predicts  obesity levels

Data obtained from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition).
""")

st.sidebar.header('User Input Features')



# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        st.sidebar.title("General Information")
        Gender = st.sidebar.selectbox('What is your gender?',('Male','Female'))
        Age = st.sidebar.slider('what is your age?', 5.00,99.99,20.00)
        Height = st.sidebar.slider('what is your height (meters)?', 0.75,2.50,1.65)
        Weight = st.sidebar.slider('what is your weight (kilograms)?', 30.0,199.0,50.00)
        st.sidebar.subheader("The attributes related with eating habits")
        family_history_with_overweight = st.sidebar.radio('Has a family member suffered or suffers from overweight?',["yes","no"],index=0,)
        FAVC = st.sidebar.radio('Do you eat high caloric food frequently (FAVC)?',["yes","no"],index=0,)
        FCVC = st.sidebar.slider('Do you usually eat vegetables in your meals (FCVC)?', 1.00,5.00,3.00)
        NCP = st.sidebar.slider('How many main meals do you have daily (NCP)?', 1.00,5.00,3.00)
        CAEC = st.sidebar.radio('Do you eat any food between meals (CAEC)?',["no","Sometimes","Frequently","Always"],index=0,)
        SMOKE = st.sidebar.radio('Do you smoke (SMOKE)?',["yes","no"],index=0,)
        st.sidebar.subheader("The attributes related with the physical condition are:")
        CH2O = st.sidebar.slider('How much water do you drink daily (litres) (CH2O)?', 1.00,5.00,3.00)
        SCC = st.sidebar.radio('Do you monitor the calories you eat daily (SCC)?',["yes","no"],index=0,)
        FAF = st.sidebar.slider('How often do you have physical activity (FAF)?', 1.00,5.00,3.00)
        TUE = st.sidebar.slider('How much time do you use technological devices such as cell phone, videogames, television, computer and others (hours) (TUE)?', 1.00,6.00,3.00)
        CALC = st.sidebar.radio('how often do you drink alcohol (CALC)?',["no","Sometimes","Frequently","Always"],index=0,) 
        MTRANS = st.sidebar.radio('Which transportation do you usually use (MTRANS)?',["Automobile","Motorbike","Bike","Public_Transportation","Walking"],index=0,)  

        data = {'Gender': Gender,
                'Age':Age,
                'Height':Height,
                'Weight':Weight,
                'family_history_with_overweight': family_history_with_overweight,
                'FAVC': FAVC,
                'FCVC': FCVC,
                'NCP': NCP,
                'CAEC': CAEC,
                'SMOKE': SMOKE,
                'CH2O': CH2O,               
                'NCP': NCP,
                'SCC': SCC,
                'FAF': FAF,
                'TUE': TUE,  
                'CALC': CALC,
                'MTRANS': MTRANS,  
                }
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

# Combines user input features with entire Obesity dataset
# This will be useful for the encoding phase
Obesity_raw = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')
Obesity = Obesity_raw.drop(columns=['NObeyesdad'])
df = pd.concat([input_df,Obesity],axis=0)

# Encoding of ordinal features
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
encode = ['Gender','family_history_with_overweight','FAVC','CAEC','SMOKE','SCC','CALC','MTRANS']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:1] # Selects only the first row (the user input data)

# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)

# Reads in saved classification model
load_clf = pickle.load(open('obesity_clf.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)


st.subheader('Prediction')
obesity_Levels = np.array(['Insufficient Weight','Normal Weight','Overweight Level I','Overweight Level II','Obesity I','Obesity II','Obesity III'])
st.write(obesity_Levels[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)
st.write("Obesity level Class Code:")
st.write("0: Insufficient Weight")
st.write("1: Normal Weight")
st.write("2: Overweight Level I")
st.write("3: Overweight Level II")
st.write("4: Obesity Type I")
st.write("5: Obesity Type II")
st.write("6: Obesity Type III")