#!/usr/bin/python3
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

data = pd.read_csv('diabetes.csv')
print(data.columns)
print(data.head(5))

data.drop('insu', axis=1, inplace=True)

print("dimension of diabetes: {}".format(data.shape))
print(data.groupby('class').size())

X_train, X_test, y_train, y_test = train_test_split(data.loc[:,data.columns!= 'class'],data['class'], stratify=data['class'], random_state=66)

#print(X_train)
#print(y_train)

KNN_model = KNeighborsClassifier(n_neighbors=5)

KNN_model.fit(X_train, y_train)
KNN_prediction = KNN_model.predict(X_test)


print(accuracy_score(KNN_prediction, y_test))

print(classification_report(KNN_prediction, y_test))


