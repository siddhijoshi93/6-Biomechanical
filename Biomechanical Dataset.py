# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 18:51:48 2021

@author: ADMIN
"""

import pandas as pd
import numpy a snp
import seaborn as sns
import matplotlib.pyplot as plt

data=pd.read_csv("C:/Users/ADMIN/Downloads/column_2C_weka.csv")

data.head()
data.columns
data.info()
data.describe()
data = data.rename({'pelvic_tilt numeric':'pelvic_tilt_numeric'}, axis = 1)
data = data.rename({'class':'classification'}, axis = 1)
#checking for missing values
data.isna().sum()

#checking for outliers
plt.boxplot(data[['pelvic_incidence', 'pelvic_tilt numeric', 'lumbar_lordosis_angle',
       'sacral_slope', 'pelvic_radius', 'degree_spondylolisthesis']])

#Data Visualaization
#checking the total abnormal and normal 
sns.countplot(data['classification'])

#plotting abnormal and normal for pelvic incidenc nd pelvic tilt
A = data[data.classification == "Abnormal"]
N = data[data.classification == "Normal"]
plt.scatter(A.pelvic_incidence, A.pelvic_tilt_numeric)
plt.scatter(N.pelvic_incidence, N.pelvic_tilt_numeric)
plt.xlabel("pelvic_incidence")
plt.ylabel("pelvic_tilt_numeric")
plt.show()

#plotting abnormal and normal for lumbar and sacral
plt.scatter(A.lumbar_lordosis_angle, A.sacral_slope)
plt.scatter(N.lumbar_lordosis_angle, N.sacral_slope)
plt.xlabel("lumbar_lordosis_angle")
plt.ylabel("sacral_slope")
plt.show()

#converting cllassification data in int
data.classification = [1 if each == "Normal" else 0 for each in data.classification]
data.classification

#model making
y=data.classification.values
x= data.drop(['classification'], axis=1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 1)

#logistic Regression'
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

#model fit
lr.fit(x_train, y_train)
lr.fit(x_train, y_train)
lr.score(x_test, y_test)

#accuracy = 0.8709677419354839

#knn classification
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
knn.score(x_test, y_test)

#accuracy= 0.8602150537634409

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
dt.score(x_test, y_test)
    
#accuracy=0.7956989247311828

#Conclusion- Logistic Regression and KNN Classification have higher accurcay for the data as compared to Descion Tree
























