import numpy as np
import copy
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from utils import sorted_v
from scipy.spatial.distance import squareform
from scipy.linalg import hankel
from scipy.cluster.hierarchy import dendrogram

from scipy.cluster.hierarchy import ClusterWarning
from warnings import simplefilter
simplefilter("ignore", ClusterWarning)



def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

def check_lags(lx, lags):
    if max(lags) >= lx:
        raise ValueError("lags must be less than the sample length.")


def demean_col(z, x, j, demean=True):
    T = z.dtype  # Assuming z is a NumPy array
    m = x.shape[0]
    assert m == len(z)

    b = m * (j - 1)
    if demean:
        s = np.sum(x[b:b + m])
        mv = s / m
        for i in range(m):
            z[i] = x[b + i] - mv
    else:
        z[:] = x[b:b + m]

    return z




def get_cross_corr(sspikes, lencorr = 30, bNorm = False):
    """
    Compute cross correlation lagged across 'lencorr' time bins between columns of 'sspikes' 
    Normalize if necessary    
    """
    sspikes = np.sqrt(sspikes)
    lenspk,num_neurons = np.shape(sspikes)
    crosscorrs = np.zeros((num_neurons, num_neurons, lencorr))
    for i in range(num_neurons):
        #spk_tmp0 = np.concatenate((sspikes[:,i].copy(), np.zeros(lencorr)))
        #spk_tmp = np.array([np.roll(spk_tmp0, t1)[:lenspk] for t1 in range(lencorr)])
        spk_tmp = hankel(sspikes[:,i],np.full(lencorr,0)).T

        crosscorrs[i,:,:] = np.dot(spk_tmp, sspikes).T   
    if bNorm:
        norm_spk = np.ones((num_neurons))
        for i in range(num_neurons):
            norm_spk[i] = np.sum(np.square(sspikes[:,i]))
        for i in range(num_neurons):
            crosscorrs[i,:,:] /= (norm_spk[i]*norm_spk[:, np.newaxis])
    return crosscorrs

def get_xcorr_dist(crosscorrs):
    num_neurons = len(crosscorrs)
    crosscorrs1 = np.zeros((num_neurons,num_neurons))    
    for i1 in range(num_neurons):
        for j1 in np.arange(i1+1, num_neurons):
            a = crosscorrs[i1,j1,:]
            b = crosscorrs[j1,i1,:]
            c = np.concatenate((a,b))
            d = np.max(c)
            if np.abs(d) > 0.00000001:
                crosscorrs1[i1,j1] =  (np.min(c)/np.max(c))
            else:
                crosscorrs1[i1,j1] = 0
                
            crosscorrs1[j1,i1] = crosscorrs1[i1,j1]
    vals = np.unique(crosscorrs1)
    return crosscorrs1


def _crossdot(x, y, lx, l):
    if l >= 0:
        return np.dot(x[:lx-l], y[1+l:])
    else:
        return np.dot(x[1-l:], y[:lx+l])

def unique(list1):
 
    # insert the list to the set
    list_set = set(list1)
    # convert the set to the list
    unique_list = (list(list_set))
    return unique_list

def cluster_cor(x,n_lencor = 30, n_clusters = 6, smaller = True, metric = "euclidean", linkage = "average"):
    
    crosscorrs = get_cross_corr(x.T, lencorr = n_lencor)
    
    
    crosscorrs = get_xcorr_dist(crosscorrs)
    fig = plt.figure(figsize=(8, 8))
    plt.imshow(crosscorrs,interpolation='nearest', aspect='auto')
    plt.colorbar() 
    plt.show()

    y = np.array([True for _ in range(x.shape[1])])
    if smaller:
        y = np.sum(crosscorrs != False,1) != 0
        crosscorrs = crosscorrs[y,:][:,y]
        #T = T[:,y]
    
    
    #cor = cross_corr_dist(T,n_lencor = 100 )
    crosscorrs[np.isnan(crosscorrs)] = 0
    
    
    crosscorrs = 1 - np.corrcoef(crosscorrs)
    #Forcing symmetric
    crosscorrs = np.tril(crosscorrs) + np.tril(crosscorrs).T
    np.fill_diagonal(crosscorrs, 0, wrap=False)
    #return crosscorrs
    
    #hierarchical_cluster = AgglomerativeClustering(n_clusters=None, metric=metric, linkage=linkage)
    model = AgglomerativeClustering(distance_threshold=n_clusters, n_clusters = None, metric = metric, linkage = linkage)
    model = model.fit(crosscorrs)
    
    
    labels = model.labels_
    fig = plt.figure(figsize=(8, 8))
    plot_dendrogram(model)#, truncate_mode="level", p=3)
    plt.show()

    a = [(l,sum(labels == l)) for l in np.sort(np.unique(labels))]
    print(a)
    threshold = 30
    for i in np.sort(np.unique(labels)):
        if sum(labels== i ) <threshold:
            labels[labels == i] = 99
    a = [(l,sum(labels == l)) for l in np.sort(np.unique(labels))]
    print(a)
    
    v = sorted_v(labels)
    
    fig = plt.figure(figsize=(8, 8))
    plt.imshow(crosscorrs[v,:][:,v], aspect='auto')
    plt.colorbar() 
    plt.show()
    return labels, y