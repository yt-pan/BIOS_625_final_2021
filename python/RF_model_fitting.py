#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load

pwd = '/home/ytpan/biostat625/final/'


if __name__ == '__main__':

    df2 = pd.read_table("/home/ytpan/biostat625/final/add_catdelay_csv/train.csv",sep=',', header=0)
    Var = ["Year","Month","DayofMonth","DayOfWeek","CRSDepTime","CRSArrTime","UniqueCarrier", "Origin","Dest", "Distance","depdelayC"]
    df2 = df2[Var]
    df2 = pd.get_dummies(df2, drop_first=False,columns=["Year","Month","DayofMonth","DayOfWeek","UniqueCarrier", "Origin", "Dest"])

    print("dataset reading complete!")

    features = df2.columns.values.tolist()
    features.remove("depdelayC")
    X0 = df2[features]
    X0 = X0.to_numpy()
    Y0 = np.array([int(y) for y in df2["depdelayC"].tolist()])

    
    print("start model fitting")

    rf2 = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_leaf=5, 
                                criterion="gini", min_samples_split=125, min_weight_fraction_leaf=0.0, 
                                max_features="auto", max_leaf_nodes=None, bootstrap=True,
                                oob_score=True, verbose=0, warm_start=False, random_state=0, 
                                n_jobs=-1, class_weight="balanced_subsample")
    rf2.fit(X0, Y0)
    print("model fitting complete")
    
    dump(rf2, pwd+'rf1215-all-withd.joblib') 
    
    print("OK")