import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
import os
def label_format(labels):
    uniq = np.unique(labels)
    tokens = dict(zip(*[np.arange(uniq.shape[0]), uniq]))
    for k, i in tokens.items():
        idx = np.where(labels == i)[0]
        labels[idx] = k
    return labels
def load_data(path = '/home/bgutman/datasets/HCP/', nodes_idx =np.array(list(range(3)) + list(range(4,38)) + list(range(39, 70))), bad_subj_include = ['168139','361941', '257845', '671855'] ):
    '''
    load data: connectomes, smooth_2e-4 thickness, smooth_2e-4 logjac, MajVote Labels, mean area equalized 
    specify if indeed
    
    path -- localization of data
    connectome path = path + 'Connectomes/' 
    thickness path = path + 'FreeSurfer/thickness/'
    logjac path = path + 'FreeSurfer/LogJacobian/'
    
    
    mean_area and labels should locate in directory path
    
    
    nodes_idx -- each connectome can be contane more than 68 vertex, so be sure you know which one to use.
    '''
    
    data = pd.DataFrame(columns=['subject_id', 'connectome', 'thickness', 'logjac'])
    path_connectomes = path + 'Connectomes/' 
    idx = nodes_idx
    bad_subj = []
    for foldername in sorted(os.listdir(path_connectomes)):
    #896778_NxNmatrix_FULL_618566.txt  896778_NxNmatrix_FULL_NORM.txt
        filename = sorted(os.listdir(path_connectomes + foldername + '/con/'))[0]
        with open(path_connectomes + foldername + '/con/' + filename, 'rb') as f:
            matrix = np.loadtxt(f, dtype='float32')
            np.fill_diagonal(matrix,0)
            matrix = squareform(matrix[np.ix_(idx,idx)])
        if len(np.where(matrix.sum(axis=0) == 0)[0]) > 0:
            bad_subj += [foldername]
        subject = pd.DataFrame(data = [[foldername, matrix, None, None]], columns=data.columns)
        data = data.append(subject)
    data.index = data.subject_id
    data = data.drop(bad_subj)
    
    with open(path + 'LH_labels_MajVote.raw', 'rb') as f:
        LH_labels = np.fromfile(f, count=-1 ,dtype='float32')

    with open(path + 'RH_labels_MajVote.raw', 'rb') as f:
        RH_labels = np.fromfile(f, count=-1 ,dtype='float32')

    labels = np.concatenate([LH_labels, RH_labels], axis = 0).astype(int)
    loc_idx = np.where(labels != 0)[0]
    
    path_thickness = path + 'FreeSurfer/thickness/'
    
    for foldername in data.index:
        if not os.path.exists(path_thickness + foldername + '/'):
            data.drop(foldername)
        else:
            with open(path_thickness + foldername + '/' + 'LH_thick_smooth_2e-4.raw', 'rb') as f:
                    LH = np.fromfile(f, count=-1 ,dtype='float32')
                    LH = np.where(LH < 0, 0, LH)

            with open(path_thickness + foldername + '/' + 'RH_thick_smooth_2e-4.raw', 'rb') as f:
                    RH = np.fromfile(f, count=-1 ,dtype='float32')
                    RH = np.where(RH < 0, 0, RH)
            thick = np.concatenate([LH, RH], axis = 0)
            data.loc[foldername]['thickness'] = thick[loc_idx]
            

    path_log_jac = path + 'FreeSurfer/LogJacobian/'
    for foldername in data.index:
        if not os.path.exists(path_log_jac + foldername + '/'):
            data.drop(foldername)
        else:
            with open(path_log_jac + foldername + '/' + 'LH_LogJac_2e-4.raw', 'rb') as f:
                    LH = np.fromfile(f, count=-1 ,dtype='float32')
            with open(path_log_jac + foldername + '/' + 'RH_LogJac_2e-4.raw', 'rb') as f:
                    RH += np.fromfile(f, count=-1 ,dtype='float32')
            logjac = np.concatenate([LH, RH], axis = 0)
            data.loc[foldername]['logjac'] = logjac[loc_idx]
        
  

    with open(path + 'LH_mean_area_equalized.raw', 'rb') as f:
        LH_area= np.fromfile(f, count=-1 ,dtype='float32')

    with open(path + 'RH_mean_area_equalized.raw', 'rb') as f:
        RH_area= np.fromfile(f, count=-1 ,dtype='float32')
    mean_area = np.concatenate([LH_area, RH_area], axis = 0)
    data = data.drop(labels = ['subject_id'], axis = 1)
    
    data = data.dropna()
    
    if bad_subj_include != None:
        data = data.drop(labels = bad_subj_include, axis = 0)
    loc_labels = label_format(labels[loc_idx])
    new_mean_area = mean_area[loc_idx]
    print('Number of subjects: {}'.format(data.shape[0]))
    print('Data contains: \nthick shape {} \nlogjac shape {} \nlabels shape {} \
          \nmean_area shape {} '.format(data.thickness[0].shape, data.logjac[0].shape, 
                                        loc_labels.shape, new_mean_area.shape)) 
    
    return data, loc_labels, new_mean_area
def load_meshes_coor_tria(path='/home/bgutman/datasets/HCP/', name='_200_mean.m'):
    '''
    read and load meshes coordinates and triangles 
    return : coord, triangles
    
    #num of mesh - [L, R]
    
    '''
    LH_coord = []
    LH_tria = []
    with open(path + 'LH' + name, 'r') as f:
        f.readline()
        for line in f:
            a = line.strip('\n').split(' ')
            if a[0] == 'Vertex':
                LH_coord += [np.array([float(x) for x in a[2:-1]])]
            if a[0] == 'Face':
                LH_tria += [np.array([int(x)-1 for x in a[2:]])]
    RH_coord = []
    RH_tria = []
    with open(path +'RH' + name, 'r') as f:
        f.readline()
        for line in f:
            a = line.strip('\n').split(' ')
            if a[0] == 'Vertex':
                RH_coord += [np.array([float(x)-1 for x in a[2:-1]])]
            if a[0] == 'Face':
                RH_tria += [np.array([int(x)+len(LH_coord) - 1 for x in a[2:]])]
    coord = np.array(LH_coord + RH_coord)
    tria = np.array(LH_tria + RH_tria)
    with open(path + 'LH_labels_MajVote.raw', 'rb') as f:
        LH_labels = np.fromfile(f, count=-1 ,dtype='float32')

    with open(path + 'RH_labels_MajVote.raw', 'rb') as f:
        RH_labels = np.fromfile(f, count=-1 ,dtype='float32')

    labels = np.concatenate([LH_labels, RH_labels], axis = 0).astype(int)
    loc_idx = np.where(labels != 0)[0]

    coord = coord[loc_idx]
    unique_labels_meshes = np.unique(loc_idx)
    numeration = dict(zip(unique_labels_meshes, np.arange(len(unique_labels_meshes))))
    del_idx = set(np.where(labels == 0)[0])
    new_tria = []
    for one in tria:
        if len(del_idx.intersection(set(one))) == 0:
            new_tria +=[one]
    new_tria = np.array(new_tria)

    n = new_tria.shape[0]
    new_tria = np.array(list(map(lambda x: numeration[x], new_tria.reshape(-1)))).reshape(n,3)
    print('Meshes coordinates shape: ', coord.shape)
    print('Number of triangles of meshes: ', new_tria.shape)
    return coord, tria
def convert(data):
    return np.array([one for one in data])


def one_subj_count(thick, log_jac, mean_area):
    area = np.exp(log_jac)*mean_area
    vol = thick * area
    return area, vol
def meshes_count(data,  mean_area): 
    new_data = data.copy()
    new_data['area'] = [None]*data.shape[0]
    new_data['vol'] = [None]*data.shape[0]
    for subj in new_data.index:
        area, vol = one_subj_count(data.thickness[subj], data.logjac[subj],
                                  mean_area)
        new_data.area[subj] = area
        new_data.vol[subj] = vol
    return new_data
def global_count(data, labels, mean_area):
    nodes = np.arange(68)
    idxes = np.array([np.where(labels == node)[0] for node in nodes])

    new_data = meshes_count(data, mean_area)
    new_data['roi_area'] = [None]*data.shape[0]
    new_data['roi_vol'] = [None]*data.shape[0]
    
    for subj in new_data.index:
        roi_area = np.array([np.sum(new_data.area[subj][idx]) for idx in idxes])
        roi_vol = np.array([np.sum(new_data.vol[subj][idx]) for idx in idxes])
        new_data.roi_area[subj] = roi_area
        new_data.roi_vol[subj] = roi_vol
    return new_data
def get_data_ij(i, j, labels, all_triangles, coordinates, data):
    idx_node_i = np.where(labels == i)[0]
    idx_node_j = np.where(labels == j)[0]
    all_idx_ij = set((idx_node_i).tolist() + (idx_node_j).tolist())
    
    triangles_ij = []
    for mesh in all_triangles:
        if len(all_idx_ij.intersection(set(mesh))) == 3:
            triangles_ij += [mesh]
    triangles_ij = np.array(triangles_ij)

    unique_triangles_nodes_ij = sorted(np.unique(triangles_ij.reshape(-1)))
    #print(len(all_idx_ij), len(unique_triangles_nodes_ij))
    if len(unique_triangles_nodes_ij) < len(all_idx_ij):
        all_idx_ij = sorted(list(all_idx_ij.intersection(set(unique_triangles_nodes_ij))))
    else:
        all_idx_ij = list(all_idx_ij)
    numeration = dict(zip(unique_triangles_nodes_ij, np.arange(len(unique_triangles_nodes_ij))))
    #print(all_idx_ij[:10])
    n = triangles_ij.shape[0]

    re_triangles_ij = np.array(list(map(lambda x: numeration[x], triangles_ij.reshape(-1)))).reshape(n,3)

    coord_ij = coordinates[all_idx_ij,:]
    
    return coord_ij, re_triangles_ij, data[:, all_idx_ij]