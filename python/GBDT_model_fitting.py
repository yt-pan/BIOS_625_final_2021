#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from joblib import dump, load

pwd = '/home/ytpan/biostat625/final/'

if __name__ == '__main__':
    df2 = pd.read_table("/home/ytpan/biostat625/final/add_catdelay_csv/train.csv",sep=',', header=0)
    Var = ["Year","Month","DayofMonth","DayOfWeek","CRSDepTime","CRSArrTime","UniqueCarrier", "Origin","Dest", "Distance","depdelayC"]
    df2 = df2[Var]
    df2 = pd.get_dummies(df2, drop_first=False,columns=["Year","Month","DayofMonth","DayOfWeek","UniqueCarrier", "Origin","Dest"])
    
    print("dataset reading complete!")

    features = df2.columns.values.tolist()
    features.remove("depdelayC")
    X0 = df2[features]
    X0 = X0.to_numpy()
    Y0 = np.array([int(y) for y in df2["depdelayC"].tolist()])
    
    print("start model fitting")
    gbdtm = GradientBoostingClassifier(max_depth=20, max_features='sqrt',
                           min_samples_leaf=15, min_samples_split=400,
                           n_estimators=200, random_state=10, subsample=0.8)

    gbdtm.fit(X0, Y0)
    print("model fitting complete")

    dump(gbdtm, pwd+'GBDT1214Fall+de.joblib')       
    print("OK")