# Package Imports
import streamlit as st
from PIL import Image
import pandas as pd
import pickle
import warnings 
warnings.filterwarnings('ignore')

# Loading the diabetes prediction model
def load_model(model):
    """
        SUMMARY
            loads model from file
        ARGS
            model: the filename for the model you want to load. 
        RETURNS
            loaded_model (model) loaded model. 
    """
    with open(model, 'rb') as file:
        loaded_model = pickle.load(file)
        return loaded_model

# Load pretrained model
svm = load_model('svc_model.pkl')

# Inserting a patient function
def insert_patient(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetespedigreefunction, age):
    """
        SUMMARY:
            Runs inference on given set of data passed in as arguments.
        ARGS:
            pregnancies
            glucose 
            blood_pressure 
            skin_thickness
            insulin
            bmi
            diabetespedigreefunction, 
            age.
        RETURNS:
            prediction (str) returns a prediction that is either 0 or 1 for no diabetes predicted or diabetes predicted. 
    """
    prediction = svm.predict([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetespedigreefunction, age]]) # where we predict on the data
    if prediction == 0:
        prediction = 'no diabetes predicted'
    elif prediction ==1:
        prediction = 'diabetes predicted'
    return prediction

# Processing the CSV to read, remove outcome column, and use the data in the model
def process_csv(csv_file):
    # read the csv
    data = pd.read_csv(csv_file)
    feature_data = data.drop(["Outcome"],axis = 1)
    # predict using this dataframe
    results = svm.predict(feature_data)
    # return results. 
    return results   


# Application Introduction
st.header("Diabetes Analysis", )
header = Image.open('diabetes.jpg')
st.image(header)
st.write("This tool allows you chose from analyzing a dataset of patients or entering a new patient's information to determine if they have diabetes.")

# Batch Analysis
st.subheader("Analyze a Dataset")
diabetes_df = st.file_uploader("Drag and drop or import the CSV file.", on_change=None)
#results = process_csv(diabetes_df)
if diabetes_df is not None:
    batch = st.button("Dataset Analysis", on_click=st.text(process_csv(diabetes_df)))
else:
    st.write("Insert csv file please")

# Stream Analysis
st.subheader("Enter a New Patient")

# Argument inputs
pregnancies = st.number_input("Pregnancies", label_visibility="visible")
glucose = st.number_input('Glucose', label_visibility="visible")
blood_pressure = st.number_input('Blood Pressure', label_visibility="visible")
skin_thickness = st.number_input('Skin Thickness', label_visibility="visible")
insulin = st.number_input('Insulin', label_visibility="visible")
bmi = st.number_input('Body Mass Index', label_visibility="visible")
diabetespedigreefunction = st.number_input('Diabetes Pedigree Function', label_visibility="visible")
age = st.number_input('Age', label_visibility="visible")

# Running stream analysis
patient_outcome=insert_patient(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetespedigreefunction, age)
if insert_patient is not None:
    stream = st.button("New Patient Analysis", on_click=st.write(f"Patient Outcome: {patient_outcome}"))
else:
    st.write("Enter new patient metrics")