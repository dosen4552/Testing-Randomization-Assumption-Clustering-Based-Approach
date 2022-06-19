# Testing-Randomization-Assumption-Clustering-Based-Approach
This is an implementation of clustering approach for testing randomization assumption: https://arxiv.org/abs/2107.00815

# Example
```python

# Load Data Example
testing_data = read_csv("pscore_match_bio.csv")
testing_data = testing_data.sort_values(by=['matched_set'])
#testing_data.to_csv("pscore_match_bio_after_rank.csv")
real_data = ((testing_data.values)[:,0:10]).astype(float)
true_label =  ((testing_data.values)[:,11]).astype(float)


# Run contrained K-means clustering
clusters, centers, B = cop_kmeans_metric_learning(dataset = test_data, learn = True, diag = True)
accuracy = sum(label == clusters)/n
print("Accuracy is", accuracy)
print("P-value is ", p_value_calculation_z(accuracy, test_data, alpha = 0.05, Gamma = 1))
print("Corresponding Gamma is ",calculate_Gamma(accuracy, test_data, alpha = 0.05))

# Run GMM clustering
clusters = GMM_RA(data = test_data)
accuracy = sum(true_label == clusters)/data.shape[0]
print("Accuracy is", accuracy)
print("P-value is ", p_value_calculation_z(accuracy, test_data, alpha = 0.05, Gamma = 1))
print("Corresponding Gamma is ",calculate_Gamma(accuracy, test_data, alpha = 0.05))
```


# Reference
1.) Wagstaff, K., Cardie, C., Rogers, S., & Schrödl, S. (2001, June). Constrained k-means clustering with background knowledge. In ICML (Vol. 1, pp. 577-584).

2.) Bradley, P. S., K. P. Bennett, and Ayhan Demiriz. "Constrained k-means clustering." Microsoft Research, Redmond (2000): 1-8.

3.) Babaki, B., Guns, T., & Nijssen, S. (2014). Constrained clustering using column generation. In Integration of AI and OR Techniques in Constraint Programming (pp. 438-454). Springer International Publishing.

4.) Guns, Tias, Christel Vrain, and Khanh-Chuong Duong. "Repetitive branch-and-bound using constraint programming for constrained minimum sum-of-squares clustering." 22nd European Conference on Artificial Intelligence. 2016.

5.) metric-learn: Metric Learning Algorithms in Python, de Vazelhes et al., Journal of Machine Learning Research, 21(138):1-6, 2020.
