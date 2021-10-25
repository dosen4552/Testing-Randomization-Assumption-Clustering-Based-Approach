# Testing-Randomization-Assumption-Clustering-Based-Approach
This is an implementation of clustering approach for testing randomization assumption

# Example
import numpy
from copkmeans.cop_kmeans import cop_kmeans
input_matrix = numpy.random.rand(100, 500)
must_link = [(0, 10), (0, 20), (0, 30)]
cannot_link = [(1, 10), (2, 10), (3, 10)]
clusters, centers = cop_kmeans(dataset=input_matrix, k=5, ml=must_link,cl=cannot_link)


# Reference
1.) Wagstaff, K., Cardie, C., Rogers, S., & Schrödl, S. (2001, June). Constrained k-means clustering with background knowledge. In ICML (Vol. 1, pp. 577-584).

2.) Bradley, P. S., K. P. Bennett, and Ayhan Demiriz. "Constrained k-means clustering." Microsoft Research, Redmond (2000): 1-8.

3.) Babaki, B., Guns, T., & Nijssen, S. (2014). Constrained clustering using column generation. In Integration of AI and OR Techniques in Constraint Programming (pp. 438-454). Springer International Publishing.

4.) Guns, Tias, Christel Vrain, and Khanh-Chuong Duong. "Repetitive branch-and-bound using constraint programming for constrained minimum sum-of-squares clustering." 22nd European Conference on Artificial Intelligence. 2016.

5.) metric-learn: Metric Learning Algorithms in Python, de Vazelhes et al., Journal of Machine Learning Research, 21(138):1-6, 2020.
