import numpy as np
import scipy.interpolate
import time
import warnings
import cv2
import scipy.io
import math
import psana.xtcav.Constants
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn import metrics


def getGroups(X, num_clusters, method):
    """
    wrapper function to return cluster assignments from one of several methods
    Arguments:
      X: profiles to group
      num_clusters: number of clusters to put profiles into
      method: clustering algorithm
    Output
      opt: the optimal number of groups for this data
    """
    if method == 'hierarchical':
        groups = hierarchicalClustering(X, num_clusters)
    elif method == 'old':
        groups = oldGroupingMethod(X, num_clusters)
    elif method == 'cosine':
        groups = hierarchicalClustering(X, num_clusters, distance='cosine')
    elif method == 'kmeans':
        model = KMeans(n_clusters=num_clusters)
        model.fit(X)
        groups = model.labels_
    elif method == 'l1':
        groups = hierarchicalClustering(X, num_clusters, distance='l1')
    else:
        groups = hierarchicalClustering(X, num_clusters)
    return groups


def oldGroupingMethod(X, num_groups):
    """
    Grouping method used in old xtcav code
    """
    num_profiles = X.shape[0]
    shots_per_group = int(np.ceil(float(num_profiles)/num_groups))
    
    group = np.zeros(num_profiles, dtype=np.int32)       #array that will indicate which group each profile sill correspond to
    group[:]=-1                             #initiated to -1
    for g in range(num_groups):                     #For each group
        currRef=np.where(group==-1)[0]  
        if currRef.size == 0:
            continue
        currRef=currRef[0]                  #We pick the first member to be the first one that has not been assigned to a group yet

        group[currRef]=g                   #We assign it the current group

        # We calculate the correlation of the first profile to the rest of available profiles
        err = np.zeros(num_profiles, dtype=np.float64);              
        for i in range(currRef, num_profiles): 
            if group[i] == -1:
                err[i] = np.corrcoef(X[currRef,:],X[i,:])[0,1]**2;

        #The 'shots_per_group-1' profiles with the highest correlation will be also assigned to the same group
        order=np.argsort(err)            
        for i in range(0, min(shots_per_group-1, len(order))): 
            group[order[-(1+i)]]=g
    return group


def hierarchicalClustering(X, num_clusters, distance='euclidean'):
    """
    wrapper function for sklearn agglomerative clustering algorithm
    """
    linkage = 'ward' if distance == 'euclidean' else 'average'
    model = AgglomerativeClustering(n_clusters=num_clusters, linkage=linkage, affinity=distance)
    model.fit(X)
    return model.labels_


def findOptGroups(X, max_num, method='hierarchical', B=30, use_SVD=True):
    """
    Helper function to find optimal # of groups for profiles using the Gap Statistic
    Arguments:
      X: profiles to group
      B: number of reference groups to generate
      max_num: maximum number of groups allowed
    Output
      opt: the optimal number of groups for this data
    """
    num_profiles, t = X.shape
    rand_cluster_variance = {}
    true_cluster_variance = {}
    sd = {}

    if use_SVD:
        #use the SVD of profiles to cluster. Speeds things up a lot...
        num_features = max(30, max_num) # use minimum of 30 features
        u, s, vt = np.linalg.svd(X.T)
        W = u[:, 0:num_features - 1]
        X = np.matmul(X, W)

    #Use svd of centered profiles to create reference sets
    column_mean = np.mean(X, axis=0)
    centered = X - column_mean
    u, s, vt = np.linalg.svd(centered)
    x_ = np.matmul(centered, vt.T)
    bounding_box = getBoundingBox(x_)
    
    reference_sets = []
    for i in range(B):
        rand_sample = generateRandSample(bounding_box, num_profiles)
        rand_sample = np.matmul(rand_sample, vt) + column_mean
        reference_sets.append(rand_sample)

    gap_statistic = {}
    sd = {}
    
    min_clusters = 2
    step = 1 if max_num - min_clusters <= 15 else 2 if max_num - min_clusters <= 30 else 3 #choose step size of 1, 2 or 3
    clusters = list(range(min_clusters+step, max_num+step, step))
    gap_statistic[min_clusters], _ = calculateGapStatistic(min_clusters, X, reference_sets, method=method)
    
    for clus in clusters:
        gap_statistic[clus], sd[clus] = calculateGapStatistic(clus, X, reference_sets, method=method)
        if gap_statistic[clus] - sd[clus]*step < gap_statistic[clus-step]:
            return clus-step
    return max_num


def calculateGapStatistic(n, X, reference_sets, method='hierarchical'):
    """
    Calculation of gap statistic for specific number of clusters
    https://statweb.stanford.edu/~gwalther/gap

    """
    B = len(reference_sets)
    groups = getGroups(X, n, method=method)
    true_cluster_variance = np.log(calculateClusterVariance(groups, X, n))
    rand_variance = []
    num_profiles = X.shape[0]
    #fit to B random reference datasets
    for k in range(B):        
        groups = getGroups(reference_sets[k], n, method=method)
        rand_variance.append(np.log(calculateClusterVariance(groups, reference_sets[k], n)))
    rand_cluster_variance = np.mean(rand_variance)
    sd = np.std(rand_variance)* np.sqrt(1+1./B)
    gap_statistic = rand_cluster_variance - true_cluster_variance
    return gap_statistic, sd


def calculateClusterVariance(assignments, data, num_clusters):
    """
    Calculation of intercluster variance
    """
    d = 0
    for group in range(num_clusters):
        points = data[assignments == group,:]
        center = np.mean(points, axis = 0)
        d += sum(np.apply_along_axis(lambda x: np.linalg.norm(x - center)**2, 1, points))
    return d

def getPercentile(data, percentile=0.9):
    a = np.cumsum(data, axis=0)
    out = np.divide(a, np.sum(data, axis=0), out=np.zeros_like(a), where=np.sum(data, axis=0)!=0)
    test = (out > 1-percentile).argmax(axis=0)
    return test
    
def trimImg(x):
    rows = np.any(x, axis=1)
    cols = np.any(x, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return x[ymin:ymax+1, xmin:xmax+1]


def getNorthCoast(imgs):
    trimmed = np.array([trimImg(f) for f in imgs])
    out = np.array([getPercentile(x) for x in trimmed])
    arrlens = np.array([len(x) for x in out])
    max_len = np.amax(arrlens)
    maxes = [np.max(x) for x in out]
    max_val = np.amax(maxes)
    def padArray(x):
        return np.pad(x, pad_width=((max_len - len(x))/2, int(np.ceil(float(max_len - len(x))/2)) ) , mode="constant", constant_values=max_val+1) 
    pad = [padArray(x) for x in out]
    return np.vstack(pad)


def generateRandSample(bounding_box, num_profiles):
    """
    generates a random sample of the same structure as the input data
    """
    return np.apply_along_axis(lambda l : np.random.uniform(l[0], l[1], num_profiles), 1, bounding_box).T


def getBoundingBox(X):
    return [(min(X[:,i]), max(X[:,i])) for i in range(X.shape[1])]


