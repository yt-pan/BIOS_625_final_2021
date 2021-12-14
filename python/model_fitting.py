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

pwd = '/home/ytpan/biostat625/final/'


if __name__ == '__main__':
    np.random.seed(1000)   
    df2 = pd.read_table("/home/ytpan/biostat625/final/add_catdelay_csv/test2.csv",sep=',', header=0)
    Var = ["Year","Month","DayofMonth","DayOfWeek","CRSDepTime","CRSArrTime","UniqueCarrier", "Origin","Dest","Distance","depdelayC"]
    df2 = df2[Var]
    df2 = pd.get_dummies(df2, drop_first=False,columns=["Year","Month","DayofMonth","DayOfWeek","UniqueCarrier", "Origin","Dest"])
    
    test_fraction = 0.30  # Fraction of total data set used for testing.

    features = df2.columns.values.tolist()
    features.remove("depdelayC")
    X0 = df2[features]
    X0 = X0.to_numpy()
    Y0 = np.array([int(y) for y in df2["depdelayC"].tolist()])

    X_train, X_test, y_train, y_test = train_test_split(X0, Y0, test_size=test_fraction, random_state=0)
    print("Training set size: %i, test set size: %i, total: %i, test fraction: %f" \
      %(len(y_train),len(y_test),len(Y0),float(len(y_test))/len(Y0)))
    

    parameter_grid = [{'n_estimators': [100, 150, 200, 250, 300]}]

    # Start the grid search
    clf1 = GridSearchCV(RandomForestClassifier(criterion="gini", min_samples_split=2, min_weight_fraction_leaf=0.0,
                                                max_features="auto", max_leaf_nodes=None, bootstrap=True,
                                                oob_score=True, verbose=0, warm_start=False, random_state=0,
                                                n_jobs=-1, class_weight="balanced_subsample",
                                                max_depth=20), 
                        parameter_grid, refit=True, cv=5, scoring='roc_auc')
    clf1.fit(X_train, y_train)

    # Print out the results
    print("Best parameter values found on development set:")
    print(" ")
    print(clf1.best_params_)
    print(" ")
    print("Grid scores on development set:")
    print(" ")
    print(clf1.cv_results_)


    parameter_grid = [{'max_depth': [10, 20, 30, 40, 50]}]

    # Start the grid search
    clf2 = GridSearchCV(RandomForestClassifier(criterion="gini", min_samples_split=50, min_weight_fraction_leaf=0.0,
                                                max_features="auto", max_leaf_nodes=None, bootstrap=True,
                                                oob_score=True, verbose=0, warm_start=False, random_state=0,
                                                n_jobs=-1, class_weight="balanced_subsample",
                                                n_estimators=clf1.best_estimator_["n_estimators"]), 
                        parameter_grid, refit=True, cv=5, scoring='roc_auc')
    clf2.fit(X_train, y_train)

    # Print out the results
    print("Best parameter values found on development set:")
    print(" ")
    print(clf2.best_params_)
    print(" ")
    print("Grid scores on development set:")
    print(" ")
    print(clf2.cv_results_)


    parameter_grid = [{'min_samples_split':[50, 75, 100, 125, 150], 'min_samples_leaf':[2,3,5,10,15]}]

    # Start the grid search
    clf3 = GridSearchCV(RandomForestClassifier(criterion="gini", min_samples_split=2, min_weight_fraction_leaf=0.0,
                                                max_features="auto", max_leaf_nodes=None, bootstrap=True,
                                                oob_score=True, verbose=0, warm_start=False, random_state=0,
                                                n_jobs=-1, class_weight="balanced_subsample",
                                                n_estimators=clf1.best_estimator_["n_estimators"],
                                                max_depth=clf2.best_estimator_["max_depth"]), 
                        parameter_grid, refit=True, cv=5, scoring='roc_auc')
    clf3.fit(X_train, y_train)

    # Print out the results
    print("Best parameter values found on development set:")
    print(" ")
    print(clf3.best_params_)
    print(" ")
    print("Grid scores on development set:")
    print(" ")
    print(clf3.cv_results_)


    # Retrieve best random forest model from grid search
    
    brf = clf3.best_estimator_
    dump(brf, pwd+'brf1213.joblib') 
    
    print("\nBest estimator:\n%s" %brf)
      
    print("OK")