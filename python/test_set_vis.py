#!/usr/bin/env python3

import gc
import pandas as pd
import numpy as np
from sklearn import metrics
from joblib import dump, load
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
ModelPath = '/home/ytpan/biostat625/final/'

if __name__ == '__main__':

    Ycol = ['r']
    Ncol = ["g"]
    print("### Test set reading Starts ###")
    df0 = pd.read_table("/home/ytpan/biostat625/final/add_catdelay_csv/train.csv",sep=',', header=0)
    df1 = pd.read_table("/home/ytpan/biostat625/final/add_catdelay_csv/test.csv",sep=',', header=0)
    df0 = pd.concat([df0,df1],keys=['s0','s1'])
    Var = ["Year","Month","DayofMonth","DayOfWeek","CRSDepTime","CRSArrTime","UniqueCarrier", "Origin","Dest","Distance","depdelayC"]
    df0 = df0[Var]
    df0 = pd.get_dummies(df0, drop_first=False,columns=["Year","Month","DayofMonth","DayOfWeek","UniqueCarrier", "Origin","Dest"])
    
    features = df0.columns.values.tolist()
    features.remove("depdelayC")
    features = ['CRSDepTime', 'CRSArrTime', 'Distance', 'Year_2003', 'Year_2004', 'Year_2005', 'Year_2006', 'Year_2007', 'Year_2008', 'Month_1', 'Month_2', 'Month_3', 'Month_4', 'Month_5', 'Month_6', 'Month_7', 'Month_8', 'Month_9', 'Month_10', 'Month_11', 'Month_12', 'DayofMonth_1', 'DayofMonth_2', 'DayofMonth_3', 'DayofMonth_4', 'DayofMonth_5', 'DayofMonth_6', 'DayofMonth_7', 'DayofMonth_8', 'DayofMonth_9', 'DayofMonth_10', 'DayofMonth_11', 'DayofMonth_12', 'DayofMonth_13', 'DayofMonth_14', 'DayofMonth_15', 'DayofMonth_16', 'DayofMonth_17', 'DayofMonth_18', 'DayofMonth_19', 'DayofMonth_20', 'DayofMonth_21', 'DayofMonth_22', 'DayofMonth_23', 'DayofMonth_24', 'DayofMonth_25', 'DayofMonth_26', 'DayofMonth_27', 'DayofMonth_28', 'DayofMonth_29', 'DayofMonth_30', 'DayofMonth_31', 'DayOfWeek_1', 'DayOfWeek_2', 'DayOfWeek_3', 'DayOfWeek_4', 'DayOfWeek_5', 'DayOfWeek_6', 'DayOfWeek_7', 'UniqueCarrier_9E', 'UniqueCarrier_AA', 'UniqueCarrier_AQ', 'UniqueCarrier_AS', 'UniqueCarrier_B6', 'UniqueCarrier_CO', 'UniqueCarrier_DH', 'UniqueCarrier_DL', 'UniqueCarrier_EV', 'UniqueCarrier_F9', 'UniqueCarrier_FL', 'UniqueCarrier_HA', 'UniqueCarrier_HP', 'UniqueCarrier_MQ', 'UniqueCarrier_NW', 'UniqueCarrier_OH', 'UniqueCarrier_OO', 'UniqueCarrier_TZ', 'UniqueCarrier_UA', 'UniqueCarrier_US', 'UniqueCarrier_WN', 'UniqueCarrier_XE', 'UniqueCarrier_YV', 'Origin_ABE', 'Origin_ABI', 'Origin_ABQ', 'Origin_ABY', 'Origin_ACK', 'Origin_ACT', 'Origin_ACV', 'Origin_ACY', 'Origin_ADK', 'Origin_ADQ', 'Origin_AEX', 'Origin_AGS', 'Origin_AKN', 'Origin_ALB', 'Origin_ALO', 'Origin_AMA', 'Origin_ANC', 'Origin_APF', 'Origin_ASE', 'Origin_ATL', 'Origin_ATW', 'Origin_AUS', 'Origin_AVL', 'Origin_AVP', 'Origin_AZO', 'Origin_BDL', 'Origin_BET', 'Origin_BFF', 'Origin_BFL', 'Origin_BGM', 'Origin_BGR', 'Origin_BHM', 'Origin_BIL', 'Origin_BIS', 'Origin_BJI', 'Origin_BLI', 'Origin_BMI', 'Origin_BNA', 'Origin_BOI', 'Origin_BOS', 'Origin_BPT', 'Origin_BQK', 'Origin_BQN', 'Origin_BRO', 'Origin_BRW', 'Origin_BTM', 'Origin_BTR', 'Origin_BTV', 'Origin_BUF', 'Origin_BUR', 'Origin_BWI', 'Origin_BZN', 'Origin_CAE', 'Origin_CAK', 'Origin_CDC', 'Origin_CDV', 'Origin_CEC', 'Origin_CHA', 'Origin_CHO', 'Origin_CHS', 'Origin_CIC', 'Origin_CID', 'Origin_CKB', 'Origin_CLD', 'Origin_CLE', 'Origin_CLL', 'Origin_CLT', 'Origin_CMH', 'Origin_CMI', 'Origin_CMX', 'Origin_COD', 'Origin_COS', 'Origin_CPR', 'Origin_CRP', 'Origin_CRW', 'Origin_CSG', 'Origin_CVG', 'Origin_CWA', 'Origin_CYS', 'Origin_DAB', 'Origin_DAL', 'Origin_DAY', 'Origin_DBQ', 'Origin_DCA', 'Origin_DEN', 'Origin_DFW', 'Origin_DHN', 'Origin_DLG', 'Origin_DLH', 'Origin_DRO', 'Origin_DSM', 'Origin_DTW', 'Origin_DUT', 'Origin_EAU', 'Origin_EFD', 'Origin_EGE', 'Origin_EKO', 'Origin_ELM', 'Origin_ELP', 'Origin_ERI', 'Origin_EUG', 'Origin_EVV', 'Origin_EWN', 'Origin_EWR', 'Origin_EYW', 'Origin_FAI', 'Origin_FAR', 'Origin_FAT', 'Origin_FAY', 'Origin_FCA', 'Origin_FLG', 'Origin_FLL', 'Origin_FLO', 'Origin_FMN', 'Origin_FNT', 'Origin_FSD', 'Origin_FSM', 'Origin_FWA', 'Origin_GCC', 'Origin_GEG', 'Origin_GFK', 'Origin_GGG', 'Origin_GJT', 'Origin_GLH', 'Origin_GNV', 'Origin_GPT', 'Origin_GRB', 'Origin_GRK', 'Origin_GRR', 'Origin_GSO', 'Origin_GSP', 'Origin_GST', 'Origin_GTF', 'Origin_GTR', 'Origin_GUC', 'Origin_HDN', 'Origin_HHH', 'Origin_HKY', 'Origin_HLN', 'Origin_HNL', 'Origin_HOU', 'Origin_HPN', 'Origin_HRL', 'Origin_HSV', 'Origin_HTS', 'Origin_HVN', 'Origin_IAD', 'Origin_IAH', 'Origin_ICT', 'Origin_IDA', 'Origin_ILE', 'Origin_ILG', 'Origin_ILM', 'Origin_IND', 'Origin_INL', 'Origin_IPL', 'Origin_ISO', 'Origin_ISP', 'Origin_ITH', 'Origin_ITO', 'Origin_IYK', 'Origin_JAC', 'Origin_JAN', 'Origin_JAX', 'Origin_JFK', 'Origin_JNU', 'Origin_KOA', 'Origin_KTN', 'Origin_LAN', 'Origin_LAS', 'Origin_LAW', 'Origin_LAX', 'Origin_LBB', 'Origin_LCH', 'Origin_LEX', 'Origin_LFT', 'Origin_LGA', 'Origin_LGB', 'Origin_LIH', 'Origin_LIT', 'Origin_LMT', 'Origin_LNK', 'Origin_LNY', 'Origin_LRD', 'Origin_LSE', 'Origin_LWB', 'Origin_LWS', 'Origin_LYH', 'Origin_MAF', 'Origin_MBS', 'Origin_MCI', 'Origin_MCN', 'Origin_MCO', 'Origin_MDT', 'Origin_MDW', 'Origin_MEI', 'Origin_MEM', 'Origin_MFE', 'Origin_MFR', 'Origin_MGM', 'Origin_MHT', 'Origin_MIA', 'Origin_MKC', 'Origin_MKE', 'Origin_MKG', 'Origin_MKK', 'Origin_MLB', 'Origin_MLI', 'Origin_MLU', 'Origin_MOB', 'Origin_MOD', 'Origin_MOT', 'Origin_MQT', 'Origin_MRY', 'Origin_MSN', 'Origin_MSO', 'Origin_MSP', 'Origin_MSY', 'Origin_MTH', 'Origin_MTJ', 'Origin_MYR', 'Origin_OAJ', 'Origin_OAK', 'Origin_OGD', 'Origin_OGG', 'Origin_OKC', 'Origin_OMA', 'Origin_OME', 'Origin_ONT', 'Origin_ORD', 'Origin_ORF', 'Origin_OTH', 'Origin_OTZ', 'Origin_OXR', 'Origin_PBI', 'Origin_PDX', 'Origin_PFN', 'Origin_PHF', 'Origin_PHL', 'Origin_PHX', 'Origin_PIA', 'Origin_PIE', 'Origin_PIH', 'Origin_PIR', 'Origin_PIT', 'Origin_PLN', 'Origin_PMD', 'Origin_PNS', 'Origin_PSC', 'Origin_PSE', 'Origin_PSG', 'Origin_PSP', 'Origin_PUB', 'Origin_PVD', 'Origin_PVU', 'Origin_PWM', 'Origin_RAP', 'Origin_RDD', 'Origin_RDM', 'Origin_RDU', 'Origin_RFD', 'Origin_RHI', 'Origin_RIC', 'Origin_RKS', 'Origin_RNO', 'Origin_ROA', 'Origin_ROC', 'Origin_ROW', 'Origin_RST', 'Origin_RSW', 'Origin_SAN', 'Origin_SAT', 'Origin_SAV', 'Origin_SBA', 'Origin_SBN', 'Origin_SBP', 'Origin_SCC', 'Origin_SCE', 'Origin_SDF', 'Origin_SEA', 'Origin_SFO', 'Origin_SGF', 'Origin_SGU', 'Origin_SHV', 'Origin_SIT', 'Origin_SJC', 'Origin_SJT', 'Origin_SJU', 'Origin_SLC', 'Origin_SLE', 'Origin_SMF', 'Origin_SMX', 'Origin_SNA', 'Origin_SOP', 'Origin_SPI', 'Origin_SPS', 'Origin_SRQ', 'Origin_STL', 'Origin_STT', 'Origin_STX', 'Origin_SUN', 'Origin_SUX', 'Origin_SWF', 'Origin_SYR', 'Origin_TEX', 'Origin_TLH', 'Origin_TOL', 'Origin_TPA', 'Origin_TRI', 'Origin_TTN', 'Origin_TUL', 'Origin_TUP', 'Origin_TUS', 'Origin_TVC', 'Origin_TWF', 'Origin_TXK', 'Origin_TYR', 'Origin_TYS', 'Origin_VCT', 'Origin_VIS', 'Origin_VLD', 'Origin_VPS', 'Origin_WRG', 'Origin_WYS', 'Origin_XNA', 'Origin_YAK', 'Origin_YKM', 'Origin_YUM', 'Dest_ABE', 'Dest_ABI', 'Dest_ABQ', 'Dest_ABY', 'Dest_ACK', 'Dest_ACT', 'Dest_ACV', 'Dest_ACY', 'Dest_ADK', 'Dest_ADQ', 'Dest_AEX', 'Dest_AGS', 'Dest_AKN', 'Dest_ALB', 'Dest_ALO', 'Dest_AMA', 'Dest_ANC', 'Dest_APF', 'Dest_ASE', 'Dest_ATL', 'Dest_ATW', 'Dest_AUS', 'Dest_AVL', 'Dest_AVP', 'Dest_AZO', 'Dest_BDL', 'Dest_BET', 'Dest_BFL', 'Dest_BGM', 'Dest_BGR', 'Dest_BHM', 'Dest_BIL', 'Dest_BIS', 'Dest_BJI', 'Dest_BLI', 'Dest_BMI', 'Dest_BNA', 'Dest_BOI', 'Dest_BOS', 'Dest_BPT', 'Dest_BQK', 'Dest_BQN', 'Dest_BRO', 'Dest_BRW', 'Dest_BTM', 'Dest_BTR', 'Dest_BTV', 'Dest_BUF', 'Dest_BUR', 'Dest_BWI', 'Dest_BZN', 'Dest_CAE', 'Dest_CAK', 'Dest_CDC', 'Dest_CDV', 'Dest_CEC', 'Dest_CHA', 'Dest_CHO', 'Dest_CHS', 'Dest_CIC', 'Dest_CID', 'Dest_CKB', 'Dest_CLD', 'Dest_CLE', 'Dest_CLL', 'Dest_CLT', 'Dest_CMH', 'Dest_CMI', 'Dest_CMX', 'Dest_COD', 'Dest_COS', 'Dest_CPR', 'Dest_CRP', 'Dest_CRW', 'Dest_CSG', 'Dest_CVG', 'Dest_CWA', 'Dest_DAB', 'Dest_DAL', 'Dest_DAY', 'Dest_DBQ', 'Dest_DCA', 'Dest_DEN', 'Dest_DFW', 'Dest_DHN', 'Dest_DLG', 'Dest_DLH', 'Dest_DRO', 'Dest_DSM', 'Dest_DTW', 'Dest_DUT', 'Dest_EAU', 'Dest_EFD', 'Dest_EGE', 'Dest_EKO', 'Dest_ELM', 'Dest_ELP', 'Dest_ERI', 'Dest_EUG', 'Dest_EVV', 'Dest_EWN', 'Dest_EWR', 'Dest_EYW', 'Dest_FAI', 'Dest_FAR', 'Dest_FAT', 'Dest_FAY', 'Dest_FCA', 'Dest_FLG', 'Dest_FLL', 'Dest_FLO', 'Dest_FNT', 'Dest_FSD', 'Dest_FSM', 'Dest_FWA', 'Dest_GCC', 'Dest_GEG', 'Dest_GFK', 'Dest_GGG', 'Dest_GJT', 'Dest_GLH', 'Dest_GNV', 'Dest_GPT', 'Dest_GRB', 'Dest_GRK', 'Dest_GRR', 'Dest_GSO', 'Dest_GSP', 'Dest_GST', 'Dest_GTF', 'Dest_GTR', 'Dest_GUC', 'Dest_HDN', 'Dest_HHH', 'Dest_HKY', 'Dest_HLN', 'Dest_HNL', 'Dest_HOU', 'Dest_HPN', 'Dest_HRL', 'Dest_HSV', 'Dest_HTS', 'Dest_HVN', 'Dest_IAD', 'Dest_IAH', 'Dest_ICT', 'Dest_IDA', 'Dest_ILE', 'Dest_ILG', 'Dest_ILM', 'Dest_IND', 'Dest_INL', 'Dest_IPL', 'Dest_ISO', 'Dest_ISP', 'Dest_ITH', 'Dest_ITO', 'Dest_IYK', 'Dest_JAC', 'Dest_JAN', 'Dest_JAX', 'Dest_JFK', 'Dest_JNU', 'Dest_KOA', 'Dest_KTN', 'Dest_LAN', 'Dest_LAS', 'Dest_LAW', 'Dest_LAX', 'Dest_LBB', 'Dest_LCH', 'Dest_LEX', 'Dest_LFT', 'Dest_LGA', 'Dest_LGB', 'Dest_LIH', 'Dest_LIT', 'Dest_LMT', 'Dest_LNK', 'Dest_LNY', 'Dest_LRD', 'Dest_LSE', 'Dest_LWB', 'Dest_LWS', 'Dest_LYH', 'Dest_MAF', 'Dest_MBS', 'Dest_MCI', 'Dest_MCN', 'Dest_MCO', 'Dest_MDT', 'Dest_MDW', 'Dest_MEI', 'Dest_MEM', 'Dest_MFE', 'Dest_MFR', 'Dest_MGM', 'Dest_MHT', 'Dest_MIA', 'Dest_MKC', 'Dest_MKE', 'Dest_MKG', 'Dest_MKK', 'Dest_MLB', 'Dest_MLI', 'Dest_MLU', 'Dest_MOB', 'Dest_MOD', 'Dest_MOT', 'Dest_MQT', 'Dest_MRY', 'Dest_MSN', 'Dest_MSO', 'Dest_MSP', 'Dest_MSY', 'Dest_MTH', 'Dest_MTJ', 'Dest_MYR', 'Dest_OAJ', 'Dest_OAK', 'Dest_OGG', 'Dest_OKC', 'Dest_OMA', 'Dest_OME', 'Dest_ONT', 'Dest_ORD', 'Dest_ORF', 'Dest_OTH', 'Dest_OTZ', 'Dest_OXR', 'Dest_PBI', 'Dest_PDX', 'Dest_PFN', 'Dest_PHF', 'Dest_PHL', 'Dest_PHX', 'Dest_PIA', 'Dest_PIE', 'Dest_PIH', 'Dest_PIR', 'Dest_PIT', 'Dest_PLN', 'Dest_PMD', 'Dest_PNS', 'Dest_PSC', 'Dest_PSE', 'Dest_PSG', 'Dest_PSP', 'Dest_PVD', 'Dest_PVU', 'Dest_PWM', 'Dest_RAP', 'Dest_RDD', 'Dest_RDM', 'Dest_RDU', 'Dest_RFD', 'Dest_RHI', 'Dest_RIC', 'Dest_RKS', 'Dest_RNO', 'Dest_ROA', 'Dest_ROC', 'Dest_ROW', 'Dest_RST', 'Dest_RSW', 'Dest_SAN', 'Dest_SAT', 'Dest_SAV', 'Dest_SBA', 'Dest_SBN', 'Dest_SBP', 'Dest_SCC', 'Dest_SCE', 'Dest_SDF', 'Dest_SEA', 'Dest_SFO', 'Dest_SGF', 'Dest_SGU', 'Dest_SHV', 'Dest_SIT', 'Dest_SJC', 'Dest_SJT', 'Dest_SJU', 'Dest_SLC', 'Dest_SLE', 'Dest_SMF', 'Dest_SMX', 'Dest_SNA', 'Dest_SOP', 'Dest_SPI', 'Dest_SPS', 'Dest_SRQ', 'Dest_STL', 'Dest_STT', 'Dest_STX', 'Dest_SUN', 'Dest_SUX', 'Dest_SWF', 'Dest_SYR', 'Dest_TEX', 'Dest_TLH', 'Dest_TOL', 'Dest_TPA', 'Dest_TRI', 'Dest_TTN', 'Dest_TUL', 'Dest_TUP', 'Dest_TUS', 'Dest_TVC', 'Dest_TWF', 'Dest_TXK', 'Dest_TYR', 'Dest_TYS', 'Dest_VCT', 'Dest_VIS', 'Dest_VLD', 'Dest_VPS', 'Dest_WRG', 'Dest_WYS', 'Dest_XNA', 'Dest_YAK', 'Dest_YKM', 'Dest_YUM']

    df1 = df0.loc["s0"]
    X0 = df0[features]
    Y0 = np.array([int(y) for y in df1["depdelayC"].tolist()])
    del df1
    gc.collect()
    X0 = X0.to_numpy()

    df0 = df0.loc["s1"]
    X1 = df0[features]
    Y1 = np.array([int(y) for y in df0["depdelayC"].tolist()])
    del df0
    gc.collect()
    X1 = X1.to_numpy()

    print("### Test set reading complete ###")

    # Retrieve best random forest model from grid search
    brf = load(ModelPath+"rf1215-all-withd.joblib")
    bgbdt = load(ModelPath+"GBDT1214Fall+de.joblib")
    
    print("### Model retriving complete ###")

    print("Random Forest:")

    print("Estimator currently in use:\n\n%s\n" %brf)

    # score the model
    Ntest    = len(Y0)
    Ntestpos = len([val for val in Y0 if val])
    NullAcc  = float(Ntest-Ntestpos)/Ntest
    #print("Mean accuracy on Training set: %s" %brf.score(X0, Y0))
    print("Mean accuracy on Test set:     %s" %brf.score(X1, Y1))
    print("Null accuracy on Test set:     %s" %NullAcc)
    print(" ")
    y_true, y_pred = Y1, brf.predict(X1)
    cm             = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:\ntn=%6d  fp=%6d\nfn=%6d  tp=%6d" %(cm[0][0],cm[0][1],cm[1][0],cm[1][1]))
    print("\nDetailed classification report: \n%s" %classification_report(y_true, y_pred))


    print('Number of test values = %i' %len(Y1))
    print('Number of test values equal to True = %i' %(len([val for val in Y1 if val])))

    # Compute and plot feature importances
    importances = brf.feature_importances_
    indices     = np.argsort(importances)[::-1]
    n_features  = min(20,indices.size)
    bins        = np.arange(n_features)
    x_labels    = np.array(features)[indices][:n_features]
    fig, axes   = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    plt.suptitle("Feature Importances-RF", fontsize=15)
    axes.bar(bins, importances[indices][:n_features], align="center", color="lightblue", alpha=0.5)
    axes.set_xticks(bins)
    axes.set_xticklabels(x_labels, ha="right", rotation=45.)
    axes.set_xlim([-0.5,bins.size-0.5])
    #plt.show()
    fig.savefig('FeatureImportances-RF.png', dpi=200, bbox_inches='tight')

    # Compute precision, recall, and queue rate as a function of threshold
    y_test_pred    = [p2 for [p1,p2] in brf.predict_proba(X1)]
    y_train_pred   = [p2 for [p1,p2] in brf.predict_proba(X0)]
    td0_precision, td0_recall, td0_thresholds = metrics.precision_recall_curve(Y1, y_test_pred)
    td0_thresholds = np.append(td0_thresholds, 1)
    n_thresholds   = td0_thresholds.size
    print('Number of thresholds = %i' %n_thresholds)
    n_max          = 100
    qr_thresholds  = np.linspace(0.0, 1.0, n_max+1)
    td0_queue_rate = []  
    for threshold in qr_thresholds:  
        td0_queue_rate.append((y_test_pred >= threshold).mean())

    # Histogram random forest output probabilities
    y_train_pred_1 = [pred for (pred,truth) in zip(y_train_pred,Y0) if truth==1]
    y_train_pred_0 = [pred for (pred,truth) in zip(y_train_pred,Y0) if truth==0]
    y_test_pred_1  = [pred for (pred,truth) in zip(y_test_pred,Y1) if truth==1]
    y_test_pred_0  = [pred for (pred,truth) in zip(y_test_pred,Y1) if truth==0]
    print("Delays in training set: %i, no-delays: %i" %(len(y_train_pred_1),len(y_train_pred_0)))
    print("Delays in test set: %i, no-delays: %i" %(len(y_test_pred_1),len(y_test_pred_0)))
    bin_edges        = np.linspace(0.0,1.0,11)
    hist1,bin_edges1 = np.histogram(y_train_pred_1, bins=bin_edges, density=False)
    hist2,bin_edges2 = np.histogram(y_train_pred_0, bins=bin_edges, density=False)
    hist1,bin_edges1 = np.histogram(y_test_pred_1, bins=bin_edges, density=False)
    hist2,bin_edges2 = np.histogram(y_test_pred_0, bins=bin_edges, density=False)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
    plt.subplots_adjust(wspace=0.40)
    axes[0].hist(y_train_pred_1, bins=20, range=[0.0,1.0], color=Ycol, histtype="step", label="delay", density=True)
    axes[0].hist(y_train_pred_0, bins=20, range=[0.0,1.0], color=Ncol, histtype="step", label="no delay", density=True)
    axes[0].set_xlabel("Classifier Output", fontsize=15)
    axes[0].legend(prop={'size': 10}, loc="upper right")
    axes[0].set_title("Training Set", fontsize=15)
    axes[1].hist(y_test_pred_1, bins=20, range=[0.0,1.0], color=Ycol, histtype="step", label="delay", density=True)
    axes[1].hist(y_test_pred_0, bins=20, range=[0.0,1.0], color=Ncol, histtype="step", label="no delay", density=True)
    axes[1].set_xlabel("Classifier Output", fontsize=15)
    axes[1].legend(prop={'size': 10}, loc="upper right")
    axes[1].set_title("Test Set", fontsize=15)
    #plt.show()
    fig.savefig('RF_probabilities.png', dpi=200, bbox_inches='tight')
        
    # Plot precision, recall, and queue rate
    fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    axis.plot(td0_thresholds, td0_precision, color="blue", label="precision")
    axis.plot(td0_thresholds, td0_recall, color="green", label="recall")
    axis.plot(qr_thresholds, td0_queue_rate, color="orange", label="queue rate")
    axis.set_xlabel("Threshold", fontsize=15)
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.grid(which="major", axis="both")
    axis.legend(prop={'size': 10}, loc="lower right")
    plt.show()
    fig.savefig("RF_PRQ.png", dpi=200, bbox_inches="tight")

    # Compute and plot ROC
    fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    fpr_test, tpr_test, thresholds   = metrics.roc_curve(Y1, y_test_pred)
    auroc_test                       = metrics.auc(fpr_test,tpr_test)
    fpr_train, tpr_train, thresholds = metrics.roc_curve(Y0, y_train_pred)
    auroc_train                      = metrics.auc(fpr_train,tpr_train)
    axis.plot(fpr_test, tpr_test, color="blue", label="test")
    axis.plot(fpr_train, tpr_train, color="orange", label="train")
    axis.plot([0.0,1.0], [0.0,1.0], 'k--')
    axis.set_xlabel("False Positive Rate", fontsize=15)
    axis.set_ylabel("True Positive Rate", fontsize=15)
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.legend(prop={'size': 10}, loc="lower right")
    #plt.show()
    fig.savefig('RF_ROC.png', dpi=200, bbox_inches='tight')

    print('Area under test ROC =  %s' %auroc_test)
    print('Area under train ROC = %s' %auroc_train)


    print("GBDT::")

    print("Estimator currently in use:\n\n%s\n" %bgbdt)

    # score the model
    Ntest    = len(Y0)
    Ntestpos = len([val for val in Y0 if val])
    NullAcc  = float(Ntest-Ntestpos)/Ntest
    #print("Mean accuracy on Training set: %s" %bgbdt.score(X0, Y0))
    print("Mean accuracy on Test set:     %s" %bgbdt.score(X1, Y1))
    print("Null accuracy on Test set:     %s" %NullAcc)
    print(" ")
    y_true, y_pred = Y1, bgbdt.predict(X1)
    cm             = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:\ntn=%6d  fp=%6d\nfn=%6d  tp=%6d" %(cm[0][0],cm[0][1],cm[1][0],cm[1][1]))
    print("\nDetailed classification report: \n%s" %classification_report(y_true, y_pred))


    print('Number of test values = %i' %len(Y1))
    print('Number of test values equal to True = %i' %(len([val for val in Y1 if val])))

    # Compute and plot feature importances
    importances = bgbdt.feature_importances_
    indices     = np.argsort(importances)[::-1]
    n_features  = min(20,indices.size)
    bins        = np.arange(n_features)
    x_labels    = np.array(features)[indices][:n_features]
    fig, axes   = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    plt.suptitle("Feature Importances-GBDT", fontsize=15)
    axes.bar(bins, importances[indices][:n_features], align="center", color="lightblue", alpha=0.5)
    axes.set_xticks(bins)
    axes.set_xticklabels(x_labels, ha="right", rotation=45.)
    axes.set_xlim([-0.5,bins.size-0.5])
    #plt.show()
    fig.savefig('FeatureImportances-GBDT.png', dpi=200, bbox_inches='tight')
    # Compute precision, recall, and queue rate as a function of threshold
    y_test_pred    = [p2 for [p1,p2] in bgbdt.predict_proba(X1)]
    y_train_pred   = [p2 for [p1,p2] in bgbdt.predict_proba(X0)]
    td0_precision, td0_recall, td0_thresholds = metrics.precision_recall_curve(Y1, y_test_pred)
    td0_thresholds = np.append(td0_thresholds, 1)
    n_thresholds   = td0_thresholds.size
    print('Number of thresholds = %i' %n_thresholds)
    n_max          = 100
    qr_thresholds  = np.linspace(0.0, 1.0, n_max+1)
    td0_queue_rate = []  
    for threshold in qr_thresholds:  
        td0_queue_rate.append((y_test_pred >= threshold).mean())
    # Histogram random forest output probabilities
    y_train_pred_1 = [pred for (pred,truth) in zip(y_train_pred,Y0) if truth==1]
    y_train_pred_0 = [pred for (pred,truth) in zip(y_train_pred,Y0) if truth==0]
    y_test_pred_1  = [pred for (pred,truth) in zip(y_test_pred,Y1) if truth==1]
    y_test_pred_0  = [pred for (pred,truth) in zip(y_test_pred,Y1) if truth==0]
    print("Delays in training set: %i, no-delays: %i" %(len(y_train_pred_1),len(y_train_pred_0)))
    print("Delays in test set: %i, no-delays: %i" %(len(y_test_pred_1),len(y_test_pred_0)))
    bin_edges        = np.linspace(0.0,1.0,11)
    hist1,bin_edges1 = np.histogram(y_train_pred_1, bins=bin_edges, density=False)
    hist2,bin_edges2 = np.histogram(y_train_pred_0, bins=bin_edges, density=False)
    hist1,bin_edges1 = np.histogram(y_test_pred_1, bins=bin_edges, density=False)
    hist2,bin_edges2 = np.histogram(y_test_pred_0, bins=bin_edges, density=False)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
    plt.subplots_adjust(wspace=0.40)
    axes[0].hist(y_train_pred_1, bins=20, range=[0.0,1.0], color=Ycol, histtype="step", label="delay", density=True)
    axes[0].hist(y_train_pred_0, bins=20, range=[0.0,1.0], color=Ncol, histtype="step", label="no delay", density=True)
    axes[0].set_xlabel("Classifier Output", fontsize=15)
    axes[0].legend(prop={'size': 10}, loc="upper right")
    axes[0].set_title("Training Set", fontsize=15)
    axes[1].hist(y_test_pred_1, bins=20, range=[0.0,1.0], color=Ycol, histtype="step", label="delay", density=True)
    axes[1].hist(y_test_pred_0, bins=20, range=[0.0,1.0], color=Ncol, histtype="step", label="no delay", density=True)
    axes[1].set_xlabel("Classifier Output", fontsize=15)
    axes[1].legend(prop={'size': 10}, loc="upper right")
    axes[1].set_title("Test Set", fontsize=15)
    #plt.show()
    fig.savefig('GBDT_probabilities.png', dpi=200, bbox_inches='tight')
        
    # Plot precision, recall, and queue rate
    fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    axis.plot(td0_thresholds, td0_precision, color="blue", label="precision")
    axis.plot(td0_thresholds, td0_recall, color="green", label="recall")
    axis.plot(qr_thresholds, td0_queue_rate, color="orange", label="queue rate")
    axis.set_xlabel("Threshold", fontsize=15)
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.grid(which="major", axis="both")
    axis.legend(prop={'size': 10}, loc="lower right")
    plt.show()
    fig.savefig("GBDT_PRQ.png", dpi=200, bbox_inches="tight")

    # Compute and plot ROC
    fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    fpr_test, tpr_test, thresholds   = metrics.roc_curve(Y1, y_test_pred)
    auroc_test                       = metrics.auc(fpr_test,tpr_test)
    fpr_train, tpr_train, thresholds = metrics.roc_curve(Y0, y_train_pred)
    auroc_train                      = metrics.auc(fpr_train,tpr_train)
    axis.plot(fpr_test, tpr_test, color="blue", label="test")
    axis.plot(fpr_train, tpr_train, color="orange", label="train")
    axis.plot([0.0,1.0], [0.0,1.0], 'k--')
    axis.set_xlabel("False Positive Rate", fontsize=15)
    axis.set_ylabel("True Positive Rate", fontsize=15)
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.legend(prop={'size': 10}, loc="lower right")
    #plt.show()
    fig.savefig('GBDT_ROC.png', dpi=200, bbox_inches='tight')

    print('Area under test ROC =  %s' %auroc_test)
    print('Area under train ROC = %s' %auroc_train)

    print("OK")
