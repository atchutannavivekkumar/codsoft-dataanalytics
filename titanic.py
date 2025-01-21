import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#load the dataset 
titanicDataSet = pd.read_csv("codsoft/Titanic-Dataset.csv")

print(titanicDataSet.head())

#getting the count of the people survived
#print(titanicDataSet['Survived'].value_counts())

#finding null values
#print(titanicDataSet.isnull().sum())

#interpolation : filling the gaps in unknown values in age column 
#with mean value

# Finding the mean of the column having NaN 
mean_value = int(np.mean(titanicDataSet['Age']))
print(mean_value)

titanicDataSet['Age'].fillna(value=mean_value, inplace=True)
  
#print(titanicDataSet.head())
#print(titanicDataSet.isnull().sum())

#droping the row from the dataset 
titanicDataSet.drop('Cabin', inplace=True, axis=1) 
# Drops rows where 'column_name' equals 'value'

print(titanicDataSet.isnull().sum())