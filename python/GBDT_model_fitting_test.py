#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from joblib import dump, load

pwd = '/home/ytpan/biostat625/final/'

if __name__ == '__main__': 
    df2 = pd.read_table("/home/ytpan/biostat625/final/add_catdelay_csv/test1.csv",sep=',', header=0)
    Var = ["Year","Month","DayofMonth","DayOfWeek","CRSDepTime","CRSArrTime","UniqueCarrier", "Origin","Dest","Distance","depdelayC"]
    df2 = df2[Var]
    df2 = pd.get_dummies(df2, drop_first=False,columns=["Year","Month","DayofMonth","DayOfWeek","UniqueCarrier", "Origin","Dest"])

    features = df2.columns.values.tolist()
    features.remove("depdelayC")
    X0 = df2[features]
    X0 = X0.to_numpy()
    Y0 = np.array([int(y) for y in df2["depdelayC"].tolist()])

    # Start the grid search1
    param_test1 = {'n_estimators':range(60,201,20)}
    gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=100,
                            min_samples_leaf=1,max_depth=10,max_features='sqrt', subsample=0.8,random_state=10), 
                            param_grid = param_test1, scoring='roc_auc',cv=5, n_jobs=-1)
    gsearch1.fit(X0, Y0)

    # Print out the results
    print("Best parameter values found on gsearch1:")
    print(" ")
    print(gsearch1.best_params_)
    print(" ")
    print("Grid scores on gsearch1:")
    print(" ")

    print(gsearch1.cv_results_)

    # Start the grid search2
    param_test2 = {'max_depth':range(5,20,2), 'min_samples_split':range(300,801,100)}
    gsearch2 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=gsearch1.best_params_["n_estimators"], min_samples_leaf=1, 
                            max_features='sqrt', subsample=0.8, random_state=10), 
                            param_grid = param_test2, scoring='roc_auc', cv=5, n_jobs=-1)
    gsearch2.fit(X0, Y0)

    # Print out the results
    print("Best parameter values found on gsearch2:")
    print(" ")
    print(gsearch2.best_params_)
    print(" ")
    print("Grid scores on gsearch2:")
    print(" ")
    print(gsearch2.cv_results_)

    # Start the grid search3
    param_test3 = {'min_samples_split':range(300,801,100), 'min_samples_leaf':range(10,20,1)}
    gsearch3 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=gsearch1.best_params_["n_estimators"], 
                            max_depth=gsearch2.best_params_["max_depth"],
                            max_features='sqrt', subsample=0.8, random_state=10), 
                            param_grid = param_test3, scoring='roc_auc', cv=5, n_jobs=-1)
    gsearch3.fit(X0, Y0)

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