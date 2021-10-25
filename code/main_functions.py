# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 14:58:41 2021

@author: chenk
"""

import random
import numpy as np
from pandas import read_csv
from metric_learn import MMC
from metric_learn import SDML
from metric_learn import ITML
import scipy
#import pandas as pd
#import time

# The main functions are modified from https://github.com/Behrouz-Babaki/COP-Kmeans

def cop_kmeans(dataset, k,B, ml=[], cl=[],
               initialization='kmpp',
               max_iter=300, tol=1e-4):

    ml, cl = transitive_closure(ml, cl, len(dataset))
    ml_info = get_ml_info(ml, dataset,B)
    tol = tolerance(tol, dataset)

    centers = initialize_centers(dataset, k, initialization,B)

    for _ in range(max_iter):
        clusters_ = [-1] * len(dataset)
        for i, d in enumerate(dataset):
            indices, _ = closest_clusters(centers, d, B)
            counter = 0
            if clusters_[i] == -1:
                found_cluster = False
                while (not found_cluster) and counter < len(indices):
                    index = indices[counter]
                    if not violate_constraints(i, index, clusters_, ml, cl):
                        found_cluster = True
                        clusters_[i] = index
                        for j in ml[i]:
                            clusters_[j] = index
                    counter += 1

                if not found_cluster:
                    return None, None

        clusters_, centers_ = compute_centers(clusters_, dataset, k, ml_info, B)
        shift = sum(l2_distance(centers[i], centers_[i], B) for i in range(k))
        if shift <= tol:
            break

        centers = centers_

    return clusters_, centers_


def cop_kmeans_with_metric(dataset, k, ml=[], cl=[], 
               initialization='kmpp', 
               max_iter=300, tol=1e-4):
    
    p = np.shape(dataset)[1]
    B = np.identity(p)
    
    ml, cl = transitive_closure(ml, cl, len(dataset))
    ml_info = get_ml_info(ml, dataset, B)
    tol = tolerance(tol, dataset)

    centers = initialize_centers(dataset, k, initialization, B)
    

    for _ in range(max_iter):
        clusters_ = [-1] * len(dataset)
        for i, d in enumerate(dataset):
            indices, _ = closest_clusters(centers, d, B)
            counter = 0
            if clusters_[i] == -1:
                found_cluster = False
                while (not found_cluster) and counter < len(indices):
                    index = indices[counter]
                    if not violate_constraints(i, index, clusters_, ml, cl):
                        found_cluster = True
                        clusters_[i] = index
                        for j in ml[i]:
                            clusters_[j] = index
                    counter += 1

                if not found_cluster:
                    return None, None, None
              
        try:
            B = metric_learner(dataset,clusters_,centers,algorithm = 'MMC')
        except:
            B = B
        clusters_, centers_ = compute_centers(clusters_, dataset, k, ml_info, B) 

        
        shift = sum(l2_distance(centers[i], centers_[i], B) for i in range(k))
        if shift <= tol:
            break

        centers = centers_
        

    return clusters_, centers_, B



def l2_distance(point1,point2, A):
    point1 = np.copy(np.array(point1))
    point2 = np.copy(np.array(point2))
    return np.dot(np.dot(point1-point2,A), (point1-point2).T ) 


def violate_constraints(data_index, cluster_index, clusters, ml, cl):
    for i in ml[data_index]:
        if clusters[i] != -1 and clusters[i] != cluster_index:
            return True

    for i in cl[data_index]:
        if clusters[i] == cluster_index:
            return True

    return False


def tolerance(tol, dataset):
    n = len(dataset)
    dim = len(dataset[0])
    averages = [sum(dataset[i][d] for i in range(n))/float(n) for d in range(dim)]
    variances = [sum((dataset[i][d]-averages[d])**2 for i in range(n))/float(n) for d in range(dim)]
    return tol * sum(variances) / dim

def closest_clusters(centers, datapoint, B):
    distances = [l2_distance(center, datapoint, B) for
                 center in centers]
    return sorted(range(len(distances)), key=lambda x: distances[x]), distances

def initialize_centers(dataset, k, method, B):
    if method == 'random':
        ids = list(range(len(dataset)))
        random.shuffle(ids)
        return [dataset[i] for i in ids[:k]]

    elif method == 'kmpp':
        chances = [1] * len(dataset)
        centers = []

        for _ in range(k):
            chances = [x/sum(chances) for x in chances]
            r = random.random()
            acc = 0.0
            for index, chance in enumerate(chances):
                if acc + chance >= r:
                    break
                acc += chance
            centers.append(dataset[index])

            for index, point in enumerate(dataset):
                cids, distances = closest_clusters(centers, point, B)
                chances[index] = distances[cids[0]]
            

        return centers
    
    elif method == 'deterministic':
        ids = list(range(len(dataset)))
        return [dataset[i] for i in ids[:k]]
        



def compute_centers(clusters, dataset, k, ml_info, B):
    cluster_ids = set(clusters)
    k_new = len(cluster_ids)
    id_map = dict(zip(cluster_ids, range(k_new)))
    clusters = [id_map[x] for x in clusters]

    dim = len(dataset[0])
    centers = [[0.0] * dim for i in range(k)]

    counts = [0] * k_new
    for j, c in enumerate(clusters):
        for i in range(dim):
            centers[c][i] += dataset[j][i]
        counts[c] += 1

    for j in range(k_new):
        for i in range(dim):
            centers[j][i] = centers[j][i]/float(counts[j])

    if k_new < k:
        ml_groups, ml_scores, ml_centroids = ml_info
        current_scores = [sum(l2_distance(centers[clusters[i]], dataset[i], B)
                              for i in group)
                          for group in ml_groups]
        group_ids = sorted(range(len(ml_groups)),
                           key=lambda x: current_scores[x] - ml_scores[x],
                           reverse=True)

        for j in range(k-k_new):
            gid = group_ids[j]
            cid = k_new + j
            centers[cid] = ml_centroids[gid]
            for i in ml_groups[gid]:
                clusters[i] = cid

    return clusters, centers

def get_ml_info(ml, dataset, B):
    flags = [True] * len(dataset)
    groups = []
    for i in range(len(dataset)):
        if not flags[i]: continue
        group = list(ml[i] | {i})
        groups.append(group)
        for j in group:
            flags[j] = False

    dim = len(dataset[0])
    scores = [0.0] * len(groups)
    centroids = [[0.0] * dim for i in range(len(groups))]

    for j, group in enumerate(groups):
        for d in range(dim):
            for i in group:
                centroids[j][d] += dataset[i][d]
            centroids[j][d] /= float(len(group))

    scores = [sum(l2_distance(centroids[j], dataset[i], B)
                  for i in groups[j])
              for j in range(len(groups))]

    return groups, scores, centroids

def transitive_closure(ml, cl, n):
    ml_graph = dict()
    cl_graph = dict()
    for i in range(n):
        ml_graph[i] = set()
        cl_graph[i] = set()

    def add_both(d, i, j):
        d[i].add(j)
        d[j].add(i)

    for (i, j) in ml:
        add_both(ml_graph, i, j)

    def dfs(i, graph, visited, component):
        visited[i] = True
        for j in graph[i]:
            if not visited[j]:
                dfs(j, graph, visited, component)
        component.append(i)

    visited = [False] * n
    for i in range(n):
        if not visited[i]:
            component = []
            dfs(i, ml_graph, visited, component)
            for x1 in component:
                for x2 in component:
                    if x1 != x2:
                        ml_graph[x1].add(x2)
    for (i, j) in cl:
        add_both(cl_graph, i, j)
        for y in ml_graph[j]:
            add_both(cl_graph, i, y)
        for x in ml_graph[i]:
            add_both(cl_graph, x, j)
            for y in ml_graph[j]:
                add_both(cl_graph, x, y)

    for i in ml_graph:
        for j in ml_graph[i]:
            if j != i and j in cl_graph[i]:
                raise Exception('inconsistent constraints between %d and %d' %(i, j))

    return ml_graph, cl_graph


def link_constraint(data):
    n = np.shape(data)[0]
    cannot_link = list(zip(np.arange(0,n,2),np.arange(1,n,2)))
    return cannot_link


def data_transformer(dataset, clusters, centers):
    n, p = np.shape(dataset)
    control = dataset[np.array(clusters) == 0,]
    treatment = dataset[np.array(clusters) == 1,]
    dissimilar = np.copy(dataset)
    #dissimilar = np.zeros((2*n, p))
    similar = np.zeros((2*n, p))
    for i in range(int(n/2)):
        similar[2*i,] = control[i,]
        similar[2*i + 1,] = centers[0]
    for i in np.arange(int(n/2),n,1):
        similar[2*i,] = treatment[i - int(n/2),]
        similar[2*i + 1,] = centers[1]
        

    
    data = np.vstack((dissimilar,similar)).reshape(( int(n/2) + n , 2 , p) )
    similar_pairs =  np.ones(n)
    dissimilar_pairs = -1* np.ones(int(n/2))
    pair_indicator = np.hstack((dissimilar_pairs,similar_pairs))
    return data, pair_indicator

def metric_learner(dataset,clusters,centers,algorithm = 'MMC',diagonal = True):
    # Metric learner based on MMC. SDML or ITML algorithm with default algorithm MMC
    data, pair_indicator = data_transformer(dataset, clusters, centers)
        
    if algorithm == 'MMC':
       mmc = MMC(diagonal=diagonal)
       mmc.fit(data, pair_indicator)
       target_matrix = mmc.A_
    elif algorithm == 'SDML':
         mmc = SDML()
         mmc.fit(data, pair_indicator)
         target_matrix = mmc.components_
    elif algorithm == 'ITML':
         mmc = ITML()
         mmc.fit(data, pair_indicator)
         target_matrix = mmc.components_
    else:
         print('Error, no such algorithm.')
         target_matrix = np.nan
    return target_matrix



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

def p_value_calculation_b(accuracy, matched_dataset, alpha = 0.05, Gamma = 1):
    n_t = int(np.shape(matched_dataset)[0]/2)
    k = n_t * accuracy
    if k < scipy.stats.binom.ppf(alpha/2, n_t, Gamma/(1+Gamma)) or k > scipy.stats.binom.ppf(1 - alpha/2, n_t, Gamma/(1+Gamma)):
       print('Reject the null. The assumption does not hold.')
    else:
       print("Fail to reject the null. The assumption holds.") 


def mean_difference(matched_data,original_data,true_label,original_label):
    true_label = np.array(true_label)
    treatment_mean = np.mean(matched_data[true_label == 1],axis = 0)
    control_mean = np.mean(matched_data[true_label == 0], axis = 0)
    treatment_var = np.var(original_data[original_label == 1],axis = 0)
    control_var = np.var(original_data[original_label == 0], axis = 0)
    return abs(treatment_mean - control_mean)/np.sqrt((treatment_var 
                                                      + control_var) / 2), np.mean(( abs(treatment_mean 
                                                                                        - control_mean)/np.sqrt( (treatment_var + control_var) / 2) ))
    

def cop_kmeans_outer_loop(dataset, no_iter = 2):
    
    n,p = np.shape(dataset)
    B = np.identity(p)
    cannot_link = link_constraint(dataset)
    
    for i in range(no_iter):
        clusters, centers = cop_kmeans(dataset=dataset, k=2, B = B,  ml=[],cl=cannot_link, initialization='deterministic',max_iter=2000,tol=1e-4)
        if i == (no_iter - 1):
           break                                                                                     
        try:
            B = metric_learner(dataset,clusters,centers,algorithm = 'MMC',diagonal = True)
        except:
            B = B                                                                              

    return clusters,centers,B


def calculate_Gamma(accuracy, matched_dataset, alpha = 0.05):
    n_t = int(np.shape(matched_dataset)[0]/2)
    k = n_t * accuracy
    for Gamma in np.arange(1,10,0.01):
    
        if k >= scipy.stats.binom.ppf(alpha/2, n_t, Gamma/(1+Gamma)) and k <= scipy.stats.binom.ppf(1 - alpha/2, n_t, Gamma/(1+Gamma)):
           return Gamma
    



############################################## Test functions #############################################################

def main_test_level(p = 10, n = 500, alpha = 0.05, level = 10000):
    # Test Level
    count = 0
    for i in range(level):
        data = np.random.normal(0,1,[n,p])
        true_label = np.array([0,1] * int(n/2))
        cannot_link = link_constraint(data)
        
        clusters, centers, B = cop_kmeans_with_metric(dataset=data, k=2, ml=[],cl=cannot_link,initialization='deterministic',max_iter=1000,tol=1e-4)
        accuracy_null = sum(true_label == clusters)/np.shape(data)[0]
        p_value_null = p_value_calculation_z(accuracy_null, data, alpha = 0.05, Gamma = 1)
        if p_value_null < alpha :
           count += 1
        print("Current number is",i)
        print("Current accuracy is",accuracy_null)
        print("Current p value is",p_value_null)
        print("Number of rejection is",count)
        print("Current level is",count/(i+1))
        
    print("Overall level is",count/level)

def main_test_real_data(real_data, true_label,Gamma = 1):
    cannot_link = link_constraint(real_data)
    n, p = np.shape(real_data)
    clusters, centers, B = cop_kmeans_with_metric(dataset=real_data, k=2, ml=[],cl=cannot_link,initialization='kmpp',max_iter=4000)
    #clusters, centers, B = cop_kmeans_outer_loop(real_data, no_iter = 2)
    accuracy = sum(true_label == clusters)/n
    p_value_calculation_b(accuracy, real_data, alpha = 0.05, Gamma = Gamma)
    print("Accuracy is", accuracy)
    #print("P value is", p_value)
    return accuracy, clusters



#########################################  Run Real Data #################################################################

random.seed(9)

# Test real dataset (bio data)

testing_data2 = read_csv("C:\\Users\\chenk\\Dropbox\\Matching and Semi-Supervised Learning\\real data all\\pscore_match_bio.csv")
testing_data2 = testing_data2.sort_values(by=['matched_set'])
testing_data2.to_csv("C:\\Users\\chenk\\Dropbox\\Matching and Semi-Supervised Learning\\real data all\\pscore_match_bio_after_rank.csv")
real_data2 = ((testing_data2.values)[:,0:10]).astype(float)
true_label =  ((testing_data2.values)[:,11]).astype(float)
accuracy, clusters = main_test_real_data(real_data2, true_label, Gamma = 1)
print("P-value is ", p_value_calculation_z(accuracy, real_data2, alpha = 0.05, Gamma = 1))
print("Corresponding Gamma is ",calculate_Gamma(accuracy, real_data2, alpha = 0.05))


















