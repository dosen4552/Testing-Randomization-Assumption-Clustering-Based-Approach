# Testing-Randomization-Assumption-Clustering-Based-Approach
This is an implementation of clustering approach for testing randomization assumption

# Example
```python
from metric_learn import MMC
from metric_learn import SDML
from metric_learn import ITML
from copkmeans.cop_kmeans import cop_kmeans

testing_data = read_csv("pscore_match_bio.csv")
testing_data = testing_data.sort_values(by=['matched_set'])
#testing_data.to_csv("pscore_match_bio_after_rank.csv")
real_data = ((testing_data.values)[:,0:10]).astype(float)
true_label =  ((testing_data.values)[:,11]).astype(float)

accuracy, clusters = main_test_real_data(real_data, true_label, Gamma = 1)
print("P-value is ", p_value_calculation_z(accuracy, real_data, alpha = 0.05, Gamma = 1))
print("Corresponding Gamma is ",calculate_Gamma(accuracy, real_data2, alpha = 0.05))
```


# Reference
1.) Wagstaff, K., Cardie, C., Rogers, S., & Schrödl, S. (2001, June). Constrained k-means clustering with background knowledge. In ICML (Vol. 1, pp. 577-584).

2.) Bradley, P. S., K. P. Bennett, and Ayhan Demiriz. "Constrained k-means clustering." Microsoft Research, Redmond (2000): 1-8.

3.) Babaki, B., Guns, T., & Nijssen, S. (2014). Constrained clustering using column generation. In Integration of AI and OR Techniques in Constraint Programming (pp. 438-454). Springer International Publishing.

4.) Guns, Tias, Christel Vrain, and Khanh-Chuong Duong. "Repetitive branch-and-bound using constraint programming for constrained minimum sum-of-squares clustering." 22nd European Conference on Artificial Intelligence. 2016.

5.) metric-learn: Metric Learning Algorithms in Python, de Vazelhes et al., Journal of Machine Learning Research, 21(138):1-6, 2020.
