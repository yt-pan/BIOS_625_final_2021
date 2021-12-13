#!/usr/bin/env python3

import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn import metrics
import bisect  
from scipy.stats import mstats
from joblib import dump, load

pwd = 'C:\\Users\\panyt\\Documents\\GitHub\\BIOS_625_final_2021\\python\\'


if __name__ == '__main__':
    np.random.seed(1000)   
    df0 = pd.read_table("C:\\Users\\panyt\\Documents\\GitHub\\BIOS_625_final_2021\\testset\\debug.csv",sep=',', header=0)
    Var = ["Year","Month","DayofMonth","DayOfWeek","CRSDepTime","CRSArrTime","UniqueCarrier", "Origin","Dest","Distance","depdelayC"]
    df1 = df0[Var]
    df2 = pd.get_dummies(df1, drop_first=False,columns=["Year","Month","DayofMonth","DayOfWeek","UniqueCarrier", "Origin","Dest"])
    
    test_fraction = 0.30  # Fraction of total data set used for testing.

    features = df2.columns.values.tolist()
    features.remove("depdelayC")
    X0 = df2[features]
    X0 = X0.to_numpy()
    Y0 = np.array([int(y) for y in df2["depdelayC"].tolist()])
    Y0 = preprocessing.label_binarize(Y0, classes=[0, 1, 2])

    X_train, X_test, y_train, y_test = train_test_split(X0, Y0, test_size=test_fraction, random_state=0)
    print("Training set size: %i, test set size: %i, total: %i, test fraction: %f" \
      %(len(y_train),len(y_test),len(Y0),float(len(y_test))/len(Y0)))
    
    # Start the grid search1
    param_test1 = {'n_estimators':range(20,201,10)}
    gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=100,
                            min_samples_leaf=1,max_depth=10,max_features='sqrt', subsample=0.8,random_state=10), 
                            param_grid = param_test1, scoring='roc_auc',iid=False,cv=5, n_jobs=-1)
    gsearch1.fit(X_train, y_train)

    # Print out the results
    print("Best parameter values found on gsearch1:")
    print(" ")
    print(gsearch1.best_params_)
    print(" ")
    print("Grid scores on gsearch1:")
    print(" ")

    print(gsearch1.cv_results_)

    # Start the grid search2
    param_test2 = {'max_depth':range(3,20,2), 'min_samples_split':range(100,801,200)}
    gsearch2 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=gsearch1.best_params_["n_estimators"], min_samples_leaf=1, 
                            max_features='sqrt', subsample=0.8, random_state=10), 
                            param_grid = param_test2, scoring='roc_auc',iid=False, cv=5,n_jobs=-1)
    gsearch2.fit(X_train, y_train)

    # Print out the results
    print("Best parameter values found on gsearch2:")
    print(" ")
    print(gsearch2.best_params_)
    print(" ")
    print("Grid scores on gsearch2:")
    print(" ")

    # Start the grid search3
    param_test3 = {'min_samples_split':range(800,1900,200), 'min_samples_leaf':range(1,10,1)}
    gsearch3 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=gsearch1.best_params_["n_estimators"], 
                            max_depth=gsearch2.best_params_["max_depth"],
                            max_features='sqrt', subsample=0.8, random_state=10), 
                            param_grid = param_test3, scoring='roc_auc',iid=False, cv=5,n_jobs=-1)
    gsearch3.fit(X_train, y_train)

    # Print out the results
    print("Best parameter values found on gsearch3:")
    print(" ")
    print(gsearch3.best_params_)
    print(" ")
    print("Grid scores on gsearch3:")
    print(" ")

    print(gsearch3.cv_results_)







    # Retrieve best random forest model from grid search
    bGBDT = gsearch3.best_estimator_
    dump(bGBDT, pwd+'bGBDT1212.joblib') 
    
    print("\nBest estimator:\n%s" %bGBDT)
      
    print("OK")