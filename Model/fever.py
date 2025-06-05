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
fever_data=pd.read_csv('csv/fever.csv')
pd.set_option('display.max_columns',None)
print(fever_data.head())
#checking the shape
print(fever_data.shape)
#droping un-important values
fever_data=fever_data.drop(columns=['Previous_Medication','Recommended_Medication'],axis=1)
print(fever_data)
#checking missing values
print(fever_data.isnull().sum())
#checking the information about the dataset
print(fever_data.info())
#encoding categorical variable
cat_cols = fever_data.select_dtypes(include='object').columns
for col in cat_cols:
                encoder = LabelEncoder()
                fever_data[col] = encoder.fit_transform(fever_data[col])
print(fever_data)
#statistical measures
print(fever_data.describe())
#Data visualisation
# A countplot of Gender
sns.countplot(x='Gender',data=fever_data,legend='auto',)
plt.title('A countplot of the gender( Female=0 vs Male=1)') 
plt.show()
# A countplot of  Diet-type
sns.countplot(x='Diet_Type',data=fever_data,legend='auto',)
plt.title('A countplot of the Diet( vergan=1,vegetarian=2,non-vegetarian=0)') 
plt.show()
# A countplot of  Diet-type in terms of gender
sns.countplot(x='Diet_Type',hue='Gender',data=fever_data,legend='auto',)
plt.title('A countplot of the Diet-Type( vergan=1,vegetarian=2,non-vegetarian=0) against Gender( Female=0 vs Male=1)') 
plt.show()
# A countplot of Gender
sns.countplot(x='Fever_Severity',data=fever_data,legend='auto',)
plt.title('A countplot of the Fever_Outcome(Normal=2,Mild=1,High=0)') 
plt.show()
#Gender in terms of Fever_Severity
sns.countplot(x='Fever_Severity',hue='Gender',data=fever_data,legend='auto',)
plt.title('A countplot of the Fever_Outcome(Normal=2,Mild=1,High=0)in terms  of Gender') 
plt.show()
#and others
#splitting the data
x=fever_data.drop(columns=['Fever_Severity'],axis=1)
y=fever_data['Fever_Severity']
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
joblib.dump(Model,'fever_disease_Prediction.pkl')
#Air quality index