import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
#loading the csv file
heart_data=pd.read_csv('csv/heart_disease.csv')
pd.set_option('display.max_columns',None)
print(heart_data.head())
#checking the shape
print(heart_data.shape)
#checking missing values
print(heart_data.isnull().sum())
print(heart_data['Gender'].value_counts())
#1.filling missing values
# Age
heart_data['Age']=heart_data['Age'].fillna(heart_data['Age'].mean())
#Gender
heart_data['Gender']=heart_data['Gender'].fillna(heart_data['Gender'].mode()[0])
#Blood Pressure
heart_data['Blood Pressure']=heart_data['Blood Pressure'].fillna(heart_data['Blood Pressure'].mean())
#Cholesterol Level
heart_data['Cholesterol Level']=heart_data['Cholesterol Level'].fillna(heart_data['Cholesterol Level'].mean())
#Exercise Habits
heart_data['Exercise Habits']=heart_data['Exercise Habits'].fillna(heart_data['Exercise Habits'].mode()[0])
# Smoking,
heart_data['Smoking']=heart_data['Smoking'].fillna(heart_data['Smoking'].mode()[0])
#droping the Family Heart Disease column
heart_data=heart_data.drop(columns=['Family Heart Disease','High Blood Pressure'],axis=1)
heart_data['Diabetes']=heart_data['Diabetes'].fillna(heart_data['Smoking'].mode()[0])
#Cholesterol Level
heart_data['BMI']=heart_data['BMI'].fillna(heart_data['BMI'].mean())
#Blood Pressure
heart_data['Blood Pressure']=heart_data['Blood Pressure'].fillna(heart_data['Blood Pressure'].mode()[0])
#Low HDL Cholesterol
heart_data['Low HDL Cholesterol']=heart_data['Low HDL Cholesterol'].fillna(heart_data['Low HDL Cholesterol'].mode()[0])
#High LDL Cholesterol,g
heart_data['High LDL Cholesterol']=heart_data['High LDL Cholesterol'].fillna(heart_data['High LDL Cholesterol'].mode()[0])
#Alcohol Consumption
heart_data['Alcohol Consumption']=heart_data['Alcohol Consumption'].fillna(heart_data['Alcohol Consumption'].mode()[0])
#Sugar Consumption'
heart_data['Sugar Consumption']=heart_data['Sugar Consumption'].fillna(heart_data['Sugar Consumption'].mode()[0])
#Stress Level
heart_data['Stress Level']=heart_data['Stress Level'].fillna(heart_data['Stress Level'].mode()[0])
#Sleep Hours
heart_data['Sleep Hours']=heart_data['Sleep Hours'].fillna(heart_data['Sleep Hours'].mean())
#Triglyceride Level
heart_data['Triglyceride Level']=heart_data['Triglyceride Level'].fillna(heart_data['Triglyceride Level'].mean())
#Fasting Blood Sugar
heart_data['Fasting Blood Sugar']=heart_data['Fasting Blood Sugar'].fillna(heart_data['Fasting Blood Sugar'].mean())
#CRP Level
heart_data['CRP Level']=heart_data['CRP Level'].fillna(heart_data['CRP Level'].mean())
#Homocysteine Level'
heart_data['Homocysteine Level']=heart_data['Homocysteine Level'].fillna(heart_data['Homocysteine Level'].mean())
print(heart_data)
#checking missing values
print(heart_data.isnull().sum())
#Encoding categorical variable
heart_data=heart_data.replace({'Gender':{'Male':1,'Female':0},'Diabetes':{'Yes':1,'No':0},
                               'Exercise Habits':{'High':2,'Medium':1,'Low':0},'Smoking':{'Yes':1,'No':0},
                               'Low HDL Cholesterol':{'Yes':1,'No':0},'High LDL Cholesterol':{'Yes':1,'No':0},
                               'Alcohol Consumption':{'High':2,'Medium':1,'Low':0},'Stress Level':{'High':2,'Medium':1,'Low':0},
                               'Sugar Consumption':{'High':2,'Medium':1,'Low':0}})
print(heart_data)
# statistical  measures
print(heart_data.describe())
#Data Visualisation
#1
sns.countplot(x='Gender',hue='Smoking',data=heart_data,palette = "Set2")
plt.title('Gender terms of Smoking')
plt.show()
#2
sns.countplot(x='Gender',hue='Diabetes',data=heart_data,palette = "Set3_r")
plt.title('Gender terms of Diabetic feature')
plt.show()
#3 More
#splitting the data
x=heart_data.drop(columns=['Heart Disease Status'],axis=1)
y=heart_data['Heart Disease Status']
#train/testing the model
#models
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print(x.shape,x_test.shape,x_train.shape)
# Random Forest Classifier
Model=RandomForestClassifier()
Model.fit(x_train,y_train)
Prediction=Model.predict(x_train)
print(f'The predicted values using  RandomForestClassifier:{Prediction}')
print(f'The accuracy score of the RandomForestClassifier model is: {accuracy_score(y_train,Prediction)}')
joblib.dump(Model,'Heart_disease_Prediction.pkl')
