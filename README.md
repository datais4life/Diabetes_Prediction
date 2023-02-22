# Machine learning models to accurately predict whether patients in the dataset have diabetes
![](/diabetes.jpg)
- Maintainer: data.is4life@gmail.com

- Environment: Dockerized Streamlit web application using Python 3.9.16 slim-buster.

## Instructions
- Download the files 
- Open the Docker application
- In a terminal, run the following code to build the Docker image:
docker build -t diabetes_app:v1 -f Dockerfile.diabetes_prediction .
- Once the image is built, run it in a container with the following code:
docker run -d -p 8501:8501 diabetes_app:v1

## Background and Purpose
This dataset is comes from the National Institute of Diabetes and Digestive and Kidney Diseases and is a selection from the Pima Indians Diabetes Database. We are using the data set with the goal of being able to predict whether a patient has diabetes or not. In particular, all patients here are females at least 21 years old of Pima Indian heritage.

The datasets consists of several medical predictor variables (X) and one target variable (y), Outcome. Predictor variables includes the number of pregnancies the patient has had, their BMI, insulin level, age, and so on.

SOURCE
https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database?resource=download

## Web Application Use
The application allows the user to either batch process a CSV of patient data or process a single patient's metrics to indicate if they have diabetes or not. The application uses a Python machine learning model, trained using Scikit-Learn's train test split, to analyze the patient's metrics and determine with a 77% accuracy if they have diabetes or not. 

![](/app.png)