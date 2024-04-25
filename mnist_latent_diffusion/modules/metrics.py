# metrics for latent space clustering
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def extent(z):
    return np.max(z, axis=0) - np.min(z, axis=0)

def extent_metric(z):
    return np.log(np.prod(extent(z)))

def density(z):
    return np.mean(np.linalg.norm(z - np.mean(z, axis=0), axis=1))

def cluster_seperation(z, y):
    cluster_seperation = 0
    
    cluster_centers = np.array([np.mean(z[y == i], axis=0) for i in range(10)])

    for i in range(10):
        cluster = z[y == i]  # pick a cluster
        distance_2_own_center = np.linalg.norm(cluster - cluster_centers[i], axis=1)
        distance_2_other_center_min = np.min([np.linalg.norm(cluster - center, axis=1) for center in cluster_centers if center is not cluster_centers[i]], axis=0)
        # print('distance_2_own_center', np.mean(distance_2_own_center))
        # print('distance_2_other_center_min', np.mean(distance_2_other_center_min))
        cluster_seperation +=  np.mean(distance_2_own_center) -  np.mean(distance_2_other_center_min)
    return cluster_seperation

# another metric is: how much empty space is there in the latent space
# we could do volumetric pixels, but this  takes for ever

def print_report(metrics):
    for k, v in metrics.items():
        print(f'{k}: {v:.2f}')

def obtain_metrics(z, y):
    return {
        'extent': extent_metric(z),
        'density': density(z),
        'cluster_seperation': cluster_seperation(z, y),
        'silhouette_score': silhouette_score(z, y),
        'davies_bouldin_score': davies_bouldin_score(z, y),
        'calinski_harabasz_score': calinski_harabasz_score(z, y),
    }

