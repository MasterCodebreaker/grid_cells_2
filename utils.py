import numpy as np
import h5py
import torch
from scipy.sparse import csc_matrix

def sorted_v(v):
    unique_values = np.unique(v)
    sorted_indices = [np.where(v == k)[0] for k in unique_values]
    flattened_indices = np.concatenate(sorted_indices)
    return flattened_indices

def cross_corr_dist(sspikes, n_lencor = 10, numcor=-1):
    
    """
    Compute cross correlation across 'lencorr' time lags between columns of 'sspikes'
    """

    sspikes = sspikes.T#.copy()
    lenspk, num_neurons = sspikes.shape
    crosscorrs = np.zeros((num_neurons, num_neurons, n_lencor))
    spk_tmp = np.zeros((lenspk, n_lencor))

    for i in range(num_neurons):
        spk_tmp0 = np.concatenate((sspikes[:, i], np.zeros(n_lencor)), axis=0)
        for ind, j in enumerate(np.arange(0,n_lencor)):
            spk_tmp[:, ind] = np.roll(spk_tmp0, j)[:lenspk]
        
        crosscorrs[i, :, :] = np.dot(spk_tmp.T, sspikes).T
    
    return cross_cor(crosscorrs)

def cross_cor(crosscorrs):
    _, num_neurons, lencorr = crosscorrs.shape
    crosscorrs1 = np.zeros((num_neurons, num_neurons))
    
    for i1 in range(num_neurons):
        for j1 in range(i1 + 1, num_neurons):
            a = crosscorrs[i1, j1, :]
            b = crosscorrs[j1, i1, :]
            c = np.concatenate((a, b))
            
            min_c = np.min(c)
            max_c = np.max(c)
            if abs(max_c) > 10e-5:
                crosscorrs1[i1, j1] = (min_c / max_c) ** 2
                crosscorrs1[j1, i1] = crosscorrs1[i1, j1]

    crosscorrs1[np.isnan(crosscorrs1)] = 0
    return crosscorrs1

def load_grid_data():
    arr = np.loadtxt("csv_files/Mouse105018_D20230802_T145627_good_cells_bool.csv",
                     delimiter=",", dtype=str)
    bool_gc = np.zeros(arr.shape[0], dtype = bool)
    
    for i in range(arr.shape[0]):
        if arr[i] == "false":
            bool_gc[i] = False
        else:
            bool_gc[i] = True

    f = h5py.File("big_files/Mouse105018_D20230802_T145627_1_5431_0.1_mouse.jld2", "r")    
    txyz_big = torch.tensor(np.array(f['single_stored_object']))
    
    f = h5py.File("big_files/Mouse105018_D20230802_T145627_1_5431_0.01_mouse.jld2", "r")    
    txyz = torch.tensor(np.array(f['single_stored_object']))
    
    f = h5py.File("sparse_matrices/Mouse105018_D20230802_T145627_big_0.1_hz_st_1_et_5431.jld", "r")    
    data = f['Mouse105018_big'][()]
    
    column_ptr=f[data[2]][:]-1 ## correct indexing from julia (starts at 1)
    indices=f[data[3]][:]-1 ## correct indexing
    values =f[data[4]][:]
    big_matrix = torch.tensor(csc_matrix((values,indices,column_ptr), shape=(data[0],data[1])).toarray()[bool_gc,:]).to_sparse()
    
    f = h5py.File("sparse_matrices/Mouse105018_D20230802_T145627_bin_0.01_hz_st_1_et_5431.jld", "r")
    f.keys()
    
    data = f['Mouse105018_bin'][()]
    
    column_ptr=f[data[2]][:]-1 ## correct indexing from julia (starts at 1)
    indices=f[data[3]][:]-1 ## correct indexing
    values =f[data[4]][:]
    bin_matrix = torch.tensor(csc_matrix((values,indices,column_ptr), shape=(data[0],data[1])).toarray()[bool_gc,:]).to_sparse()
    
        
    return bin_matrix, big_matrix, txyz, txyz_big



def load_lec_data(rat):
    if rat == "elm":
        """
        Here we choose times we are interested in
        
        #1,  sleep_box_1,        start=76,   end=5890    
        #2,  sleep_box_1,        start=6377, end=6682    
        #3,  sequence_task_1,    start=6696, end=6987    
        #4,  sleep_box_1,        start=7030, end=7332    
        #5,  sequence_task_1,    start=7347, end=7604    
        #6,  sleep_box_1,        start=7648, end=7957    
        #7,  sequence_task_1,    start=7969, end=8224    
        #8,  sleep_box_1,        start=8260, end=8577    
        """
        
        f = h5py.File("./sparse_matrices_lec/elm_2_bin_0.01_hz_st_0_et_8578.jld", "r")
        f.keys()
        data = f["elm_2_bin"][()]
        column_ptr=f[data[2]][:]-1 ## corr ect indexing from julia (starts at 1)
        indices=f[data[3]][:]-1 ## correct indexing
        values =f[data[4]][:]
        bin_matrix = torch.tensor(csc_matrix((values,indices,column_ptr), shape=(data[0],data[1])).toarray()[:,:]).to_sparse()
    
        f = h5py.File("./sparse_matrices_lec/elm_2_bin_HUGE0.01_hz_st_0_et_8578.jld", "r")
        data = f["elm_2_bin"][()]
        column_ptr=f[data[2]][:]-1 ## correct indexing from julia (starts at 1)
        indices=f[data[3]][:]-1 ## correct indexing
        values =f[data[4]][:]
        big_matrix = torch.tensor(csc_matrix((values,indices,column_ptr), shape=(data[0],data[1])).toarray()[:,:]).to_sparse()
    
        f = h5py.File("./sparse_matrices_lec/elm_2_time_loc_big_0.1_hz_st_0_et_8578.jld", "r")
        data = f["elm_2_bin"][()]
        txyz_big = torch.tensor(data)
    
        f = h5py.File("./sparse_matrices_lec/elm_2_time_loc_bin_0.01_hz_st_0_et_8578.jld", "r")
        data = f["elm_2_bin"][()]
        txyz = torch.tensor(data)
        bin_matrix = (bin_matrix.to_dense()[:, torch.isnan(txyz[1,:]) == False]).to_sparse()
        big_matrix = (big_matrix.to_dense()[:, torch.isnan(txyz_big[1,:]) == False]).to_sparse()
        txyz_big = txyz_big[:, torch.isnan(txyz_big[1,:]) == False]
        txyz = txyz[:, torch.isnan(txyz[1,:]) == False]
        
    elif rat == "hemlock_2":
        
        f = h5py.File("./sparse_matrices_lec/hemlock_2_bin_0.01_hz_st_0_et_1894.jld", "r")
        f.keys()
        data = f["hemlock_2_bin"][()]
        column_ptr=f[data[2]][:]-1 ## corr ect indexing from julia (starts at 1)
        indices=f[data[3]][:]-1 ## correct indexing
        values =f[data[4]][:]
        bin_matrix = torch.tensor(csc_matrix((values,indices,column_ptr), shape=(data[0],data[1])).toarray()[:,:]).to_sparse()
    
        f = h5py.File("./sparse_matrices_lec/hemlock_2_bin_HUGE0.01_hz_st_0_et_1894.jld", "r")
        data = f["hemlock_2_bin"][()]
        column_ptr=f[data[2]][:]-1 ## correct indexing from julia (starts at 1)
        indices=f[data[3]][:]-1 ## correct indexing
        values =f[data[4]][:]
        big_matrix = torch.tensor(csc_matrix((values,indices,column_ptr), shape=(data[0],data[1])).toarray()[:,:]).to_sparse()
    
        f = h5py.File("./sparse_matrices_lec/hemlock_2_time_loc_big_0.1_hz_st_0_et_1894.jld", "r")
        data = f["hemlock_2_bin"][()]
        txyz_big = torch.tensor(data)
    
        f = h5py.File("./sparse_matrices_lec/hemlock_2_time_loc_bin_0.01_hz_st_0_et_1894.jld", "r")
        data = f["hemlock_2_bin"][()]
        txyz = torch.tensor(data)
        
        bin_matrix = (bin_matrix.to_dense()[:, torch.isnan(txyz[1,:]) == False]).to_sparse()
        big_matrix = (big_matrix.to_dense()[:, torch.isnan(txyz_big[1,:]) == False]).to_sparse()
        txyz_big = txyz_big[:, torch.isnan(txyz_big[1,:]) == False]
        txyz = txyz[:, torch.isnan(txyz[1,:]) == False]

    return bin_matrix, big_matrix, txyz, txyz_big