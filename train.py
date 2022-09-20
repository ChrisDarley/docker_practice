import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from joblib import dump
import os

# function to run the training script and output files
def run_training():
    
    # loading iris dataset and partitioning into train and test
    X, y = load_iris(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # training a knn model on the train subset
    clf = KNeighborsClassifier()
    clf.fit(X_train, y_train)

    # dumping model along with X_test and y_test
    dump(clf, 'model.csv')
    dump(X_test, 'X_test.csv')
    dump(y_test, 'y_test.csv')

    # printing that train phase is finished
    print("Training finished")