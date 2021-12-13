#!/usr/bin/env python3

import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn import metrics
import bisect  
from scipy.stats import mstats
from joblib import dump, load

pwd = 'C:\\Users\\panyt\\Documents\\GitHub\\BIOS_625_final_2021\\python\\'

def selectSubarray( arr, elval, frac ):
    mask = np.ones(len(arr), dtype=bool)
    for ind,val in enumerate(arr):
        if val == elval:
            if np.random.uniform() <= frac:
                mask[ind] = False
    return mask


if __name__ == '__main__':
    np.random.seed(1000)   
    df0 = pd.read_table("C:\\Users\\panyt\\Documents\\GitHub\\BIOS_625_final_2021\\testset\\test.csv",sep=',', header=0)
    Var = ["Year","Month","DayofMonth","DayOfWeek","CRSDepTime","CRSArrTime","UniqueCarrier", "Origin","Dest","Distance","depdelayC"]
    df1 = df0[Var]
    df2 = pd.get_dummies(df1, drop_first=False,columns=["Year","Month","DayofMonth","DayOfWeek","UniqueCarrier", "Origin","Dest"])
    
    test_fraction     = 0.30  # Fraction of total data set used for testing.

    features = df2.columns.values.tolist()
    features.remove("depdelayC")
    X0 = df2[features]
    X0 = X0.to_numpy()
    Y0 = np.array([int(y) for y in df2["depdelayC"].tolist()])
    Y0 = preprocessing.label_binarize(Y0, classes=[0, 1, 2])

    X_train, X_test, y_train, y_test = train_test_split(X0, Y0, test_size=test_fraction, random_state=0)
    print("Training set size: %i, test set size: %i, total: %i, test fraction: %f" \
      %(len(y_train),len(y_test),len(Y0),float(len(y_test))/len(Y0)))
    

    Ycol = ["red"]
    Ncol = ["green"]

    parameter_grid = [{'n_estimators': [400, 600], 'max_depth': [40, 60], 
                    'min_samples_leaf': [3]}]

    # Start the grid search
    clf = GridSearchCV(RandomForestClassifier(criterion="gini", min_samples_split=2, min_weight_fraction_leaf=0.0,
                                            max_features="auto", max_leaf_nodes=None, bootstrap=True,
                                            oob_score=True, verbose=0, warm_start=False, random_state=0,
                                            n_jobs=-1, class_weight="balanced_subsample"), 
                    parameter_grid, refit=True, cv=3, scoring='roc_auc')
    clf.fit(X_train, y_train)

    # Print out the results
    print("Best parameter values found on development set:")
    print(" ")
    print(clf.best_params_)
    print(" ")
    print("Grid scores on development set:")
    print(" ")

    print(clf.cv_results_)

    # Retrieve best random forest model from grid search
    brf = clf.best_estimator_
    dump(brf, pwd+'brf1212.joblib') 
    
    print("\nBest estimator:\n%s" %brf)
      
    print("OK")