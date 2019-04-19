import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("all_incumbents.csv")

#print(dataset.shape)

#print(dataset.head())

X = dataset.drop(['Election Year', 'Birth Year','Name', 'Winner', 'KEY_1', 'KEY_2'], axis=1)

y = dataset['Winner']

#print(X)

from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
X = imp.fit_transform(X)

#print(X)

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

from sklearn.tree import DecisionTreeClassifier  
classifier = DecisionTreeClassifier()  
classifier.fit(X_train, y_train)  

y_pred = classifier.predict(X_test) 
#y_prob = classifier.predict_proba(X_test)

#print('Probability for the test set is: ')
#print(y_prob)

from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.metrics import precision_score
cm = confusion_matrix(y_test, y_pred)
print(cm)  
print(classification_report(y_test, y_pred)) 
print(precision_score(y_test, y_pred, average = 'weighted'))

g = 0
total = 0
while (g <= 1000):
	X = imp.fit_transform(X)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
	classifier = DecisionTreeClassifier()  
	classifier.fit(X_train, y_train)
	y_pred = classifier.predict(X_test) 
	g = g + 1
	total = total + precision_score(y_test, y_pred, average = 'weighted')

avg = total/10
print('Average accuracy: ')
print(str(round(avg, 2)) + '%')
#print(y_pred)

feat_importance = classifier.tree_.compute_feature_importances(normalize=True)
print("feat importance = " + str(feat_importance))

print('Here is a new prediction for Bernie Sanders: ')

candidate = pd.read_csv("Predict_test.csv")
prediction = classifier.predict(candidate)
#pred_prob = classifier.predict_proba(candidate)

print(prediction) 
#print(pred_prob)
print(cm[0][0])
print(cm[0][1])
print(cm[1][0])
print(cm[1][1])