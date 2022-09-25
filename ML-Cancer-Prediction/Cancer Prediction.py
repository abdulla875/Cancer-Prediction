# -*- coding: utf-8 -*-
"""
@author: abdula 
"""
#to manipulate data as a dataframe
import pandas as pd
import numpy as np
#to preprosess data
from sklearn import preprocessing
#to visualise data and results
import matplotlib.pyplot as plt
#to split data into training and test dataset
from sklearn.model_selection import train_test_split
#to use decision tree classification algorithm
from sklearn.tree import DecisionTreeClassifier
#to calculate accuracy score
from sklearn.metrics import accuracy_score
#to get confusion matrix
from sklearn.metrics import confusion_matrix
#to plot confusion matrix
from sklearn.metrics import plot_confusion_matrix
import seaborn as sns 
#to get PCA
from sklearn.decomposition import PCA
#to get GridSearchCV
from sklearn.model_selection import GridSearchCV
#to get cross_val_score
from sklearn.model_selection import cross_val_score


#https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
# The link above is for sklearn which was used to implement the model

# Import the data set 
data = pd.read_csv('data.csv')

# Check data info such as the data type of each column 
print (data.info())

# Check if there are any null value
print (data.isnull().any())

# Remove the column with the null value and the ID column as it's unnecessary

data.drop(['id', 'Unnamed: 32'], inplace=True,axis=1)

print (data.head())

# Change the disgnosis position from first to last 
new_cols = [col for col in data.columns if col != 'diagnosis'] + ['diagnosis']
data = data[new_cols]
print(data.head())

#Showing the diagnosis column had string value 
print(data["diagnosis"].unique())

#Using labelEncoder to transform the string value in diagnosis column to numeric value
label_enconder = preprocessing.LabelEncoder()
data["diagnosis"] = label_enconder.fit_transform(data["diagnosis"])
print(data.head())

#number of Benign and Malignant
diagnosisvalue = data["diagnosis"].value_counts()
print (diagnosisvalue.head())

#visualise the diagnosis using bar chart
sns.set(font_scale=1.4)
data['diagnosis'].value_counts().plot(kind='bar', figsize=(7, 6), rot=0)
plt.xlabel("Diagnosis", labelpad=14)
plt.ylabel("Number of Benign and Malignant", labelpad=14)
plt.title("Total Number of Benign and Malignant ", y=1.02);


# Visualise this distribution 
histogram  = data.hist(figsize=(20,20))

# Fuature selections
import seaborn as sns
corr = data.corr()
corr_features = corr.index
plt.figure(figsize=(20,20))
g=sns.heatmap(data[corr_features].corr(),annot=True, cmap="RdBu_r")


# Test set and the tring set
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=10)

# Feature Scaling  
from sklearn.preprocessing import StandardScaler
StandardScaler = StandardScaler()
X_train = StandardScaler.fit_transform(X_train)
X_test = StandardScaler.transform(X_test)

#https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b
#remove feature 
corrtarget = (corr["diagnosis"])
selected_features = corrtarget[corrtarget>0.4].index
data = data[selected_features]

# Feature Extraction using PCA
pca = PCA(n_components = 3)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Decision Tree Classifier Model
model_classifier = DecisionTreeClassifier()
model_classifier.fit(X_train,y_train)
target_pred= model_classifier.predict(X_test)
print("Accuracy score: {:.2f} %".format(accuracy_score(y_test,target_pred,normalize=True)*100)) 


# implement the confusion matrix
cm = confusion_matrix(y_test,target_pred)
print(cm)
accuracy_score(y_test,target_pred)
plot_confusion_matrix(model_classifier, X_test, y_test)  
plt.grid(None)
plt.show()

# cross-validation for decision tree

cv_decision_tree = cross_val_score(estimator = model_classifier,X = X_train, y = y_train, cv = 10)
print ("Cross validation Accuracy: {:.2f} %".format(cv_decision_tree.mean()*100))
plt.boxplot(cv_decision_tree)
plt.show()

#https://www.nbshare.io/notebook/312837011/Decision-Tree-Regression-With-Hyper-Parameter-Tuning-In-Python/
#tune model hyperparameters
parameters = {'criterion':["gini","entropy"],'max_depth':[1,3,5,7,9,11,12], 
              'min_samples_leaf':[5,10,20,30,40,50,100]}
gridsearch = GridSearchCV(estimator = model_classifier,
param_grid = parameters, scoring = 'accuracy', cv = 10, n_jobs = -1)
gridsearch.fit(X_train, y_train)
bestaccuracy = gridsearch.best_score_
bestparameters = gridsearch.best_params_
print("Accuracy: {:.2f} %".format(bestaccuracy*100))
print("Parameters:", bestparameters)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0)
cv_logistic_regression = cross_val_score(estimator = clf, X = X_train, y = y_train, cv = 10)
print ("Cross validation Accuracy: {:.2f} %".format(cv_logistic_regression.mean()*100))

#SVM
from sklearn.svm import SVC
svc_classifier = SVC(kernel = 'linear', random_state =0)
cv_svc = cross_val_score(estimator = svc_classifier, X = X_train, y = y_train, cv = 10)
print ("Cross validation Accuracy: {:.2f} %".format(cv_svc.mean()*100))

# Box plot
df_cross_val_score = cv_decision_tree, cv_logistic_regression,cv_svc
plt.boxplot(df_cross_val_score)
plt.show()
