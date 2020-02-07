import os
import sys
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn import tree
import pickle
from sklearn.model_selection import train_test_split
import warnings
from sklearn.svm import SVC
from sklearn import neighbors
from sklearn.neural_network import MLPClassifier

filename = '/home/sxz/data/geolife_Data/My_data_for_DL_kfold_dataset_RL.pickle'
with open(filename, 'rb') as f:
    kfold_dataset, unlabel = pickle.load(f)


# unlabel = unlabel[random_sample]
X = kfold_dataset[0][0]
Y = kfold_dataset[0][1]
random_sample = np.random.choice(len(X), size=int(1*len(X)), replace=True, p=None)
X = X[random_sample]
Y = Y[random_sample]
T_X = kfold_dataset[0][2]
T_Y = kfold_dataset[0][4]
# MLP part
# Decision Tree part
# print(X.shape)
print(Y)
print(T_Y)
# sys.exit(0)
X.resize(len(X),992)
print(X.shape)
# print(T_X.shape)
T_X.resize(len(T_X),992)
# sys.exit(0)
clf = tree.DecisionTreeClassifier()
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.3)
clf.fit(X,Y)
pred = clf.predict(T_X)
f1_weight = f1_score(T_Y, pred, average='weighted')
print(f1_weight)
print("decision tree")
print(confusion_matrix(T_Y,pred))
print(classification_report(T_Y,pred))
print(accuracy_score(T_Y,pred))


# Linear SVM

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.3)
# svclassifier = SVC(kernel = 'linear',C=0.05)
# svclassifier.fit(X_train, Y_train)
# y_pred = svclassifier.predict(X_test)
# print("Linear Support Vector Machine")
# print(confusion_matrix(Y_test,y_pred))
# print(classification_report(Y_test,y_pred))
# print(accuracy_score(Y_test,y_pred))


# Non Linear SVM

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.3)
# svclassifier = SVC(kernel = 'poly',C=0.05,degree =1,gamma='auto')
# svclassifier.fit(X_train, Y_train)
# y_pred = svclassifier.predict(X_test)
# print("Non Linear Support Vector Machine")
# print(confusion_matrix(Y_test,y_pred))
# print(classification_report(Y_test,y_pred))
# print(accuracy_score(Y_test,y_pred))


# KNN

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.3)
# knn = neighbors.KNeighborsClassifier(n_neighbors=10,weights = 'distance')
# knn.fit(X_train,Y_train)
# y_pred = knn.predict(X_test)
# print("K nearest neighbor Vector Machine")
# print(confusion_matrix(Y_test,y_pred))
# print(classification_report(Y_test,y_pred))
# print(accuracy_score(Y_test,y_pred))

# MLP
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.3)
# mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(2, 3), random_state=1)
# mlp.fit(X_train, Y_train)
# y_pred = mlp.predict(X_test)
# print("Multilayer Perceptron")
# print(confusion_matrix(Y_test,y_pred))
# print(classification_report(Y_test,y_pred))
# print(accuracy_score(Y_test,y_pred))