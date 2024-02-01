# Import necessary libraries
import numpy as np
import sklearn
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics

# Load the digits dataset
digits = load_digits()
data = digits.data
y = digits.target

# Define the number of clusters
k = 10

# Get the shape of the data
sample, features = data.shape

# Define a function to evaluate and display KMeans clustering performance
def bench_k_means(estimator, name, data):
    # Fit the KMeans estimator to the data
    estimator.fit(data)
    
    # Print clustering performance metrics
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))

# Initialize KMeans clustering with specified parameters
clf = KMeans(n_clusters=k, init="random", n_init=10)

# Evaluate and display clustering performance
bench_k_means(clf, "1", data)
