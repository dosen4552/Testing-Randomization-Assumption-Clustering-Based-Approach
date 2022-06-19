# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from pandas import read_csv
from sklearn.mixture import GaussianMixture
import numpy as np
import scipy
import pandas as pd


def GMM_RA(data):
        
    gmm = GaussianMixture(n_components=2,weights_init = [0.5,0.5],random_state = 1)
    gmm.fit(data)
    prob = gmm.predict_proba(data)
    pred_Z = np.zeros(data.shape[0])
    
    for i in np.arange(0,data.shape[0],2):
        if prob[i,0] > prob[i+1,0]:
            pred_Z[i] = 1
            pred_Z[i+1] = 0
        else:
            pred_Z[i] = 0
            pred_Z[i+1] = 1
    

    return pred_Z

def p_value_calculation_z(accuracy, matched_dataset, alpha = 0.05, Gamma = 1):
    # Conduct right-tailed test
    n_t = int(np.shape(matched_dataset)[0]/2)
    z = (1 + Gamma)*(accuracy * n_t - n_t * Gamma/(1 + Gamma))/ np.sqrt(n_t * Gamma)
    if z**2 >= scipy.stats.chi2.ppf(1 - alpha, 1):
       print("Reject the null. The assumption does not hold.")
    else:
       print("Fail to reject the null. The assumption holds.")
    p_value = 1 - scipy.stats.chi2.cdf(z**2, 1)
    return p_value

def calculate_Gamma(accuracy, matched_dataset, alpha = 0.05):
    if accuracy < 0.5:
        accuracy = 1 - accuracy
    n_t = int(np.shape(matched_dataset)[0]/2)
    k = n_t * accuracy
    for Gamma in np.arange(1,10,0.01):
    
        if k >= scipy.stats.binom.ppf(alpha/2, n_t, Gamma/(1+Gamma)) and k <= scipy.stats.binom.ppf(1 - alpha/2, n_t, Gamma/(1+Gamma)):
           return Gamma


def GMM_real_data(d = 10, file_name = 'optimal_subset_match.csv'):
    path = str('C:/Users/chenk/Dropbox/Matching and Semi-Supervised Learning/real data all/') + str(file_name)
    testing_data = read_csv(path)
    testing_data = testing_data.sort_values(by=['matched_set'])
    data = ((testing_data.values)[:,1:(1 + d)]).astype(float)
    true_label = ((testing_data.values)[:,(1 + d)]).astype(float)
    
    gmm = GaussianMixture(n_components=2,weights_init = [0.5,0.5],random_state = 1)
    gmm.fit(data)
    prob = gmm.predict_proba(data)
    pred_Z = np.zeros(data.shape[0])
    
    for i in np.arange(0,data.shape[0],2):
        if prob[i,0] > prob[i+1,0]:
            pred_Z[i] = 1
            pred_Z[i+1] = 0
        else:
            pred_Z[i] = 0
            pred_Z[i+1] = 1
    
    accuracy = sum(true_label == pred_Z)/data.shape[0]
    pval = p_value_calculation_z(accuracy, data, alpha = 0.05, Gamma = 1)
    Gamma = calculate_Gamma(accuracy, data, alpha = 0.05)
    
    result = np.vstack((true_label, pred_Z)).T
    pair = pd.DataFrame(result)
    pair = pair.rename(columns={0: 'true_label', 1: 'predicted_label'})
    pair.to_csv(path + str('result_') + str('optimal_subset_match') + str('_GMM.csv'))
            
    return pred_Z, accuracy, pval, Gamma

########################### Example ###########################

#data = read_csv('D:\\data\\simu_n_3000_d_10_C_1\\10\\opt_matched.csv')
data = read_csv('/Users/kanchen/Dropbox/Matching and Semi-Supervised Learning/simulation/all simu datasets/simu_n_5000_d_10_C_0.5/1/psm_matched.csv')
data = data.sort_values(by=['matched_set'])
test_data = ((data.values)[:,1:11]).astype(float)
true_label = ((data.values)[:,11]).astype(float)

clusters = GMM_RA(data = test_data)

accuracy = sum(true_label == clusters)/data.shape[0]
pval = p_value_calculation_z(accuracy, data, alpha = 0.05, Gamma = 1)
Gamma = calculate_Gamma(accuracy, data, alpha = 0.05)

   

