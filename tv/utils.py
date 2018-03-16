import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from collections import Counter
import networkx as nx

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





def integration_mesh_to_tria(triangles, meshes_think):
    assert triangles.shape[-1] == 3, 'Not correct traingles, should consist of 3 nodes'
    k, _ = triangles.shape
    n, _ = meshes_think.shape
    new_x = np.zeros((n, k))

    if min(triangles.reshape(-1)) != 0:
        triangles -= 1

        for i in range(k):
            new_x[:, i] = meshes_think[:, np.ix_(triangles[i])].mean(axis = -1).reshape(-1)

    return new_x

def to_degree(data, mode = 'binary'):
    '''
    data: [number of subject, square matrix] or square matrix
    convert data type of [number of subject, square matrix] -> [number of subject, degree]
    
    
    '''
    
    if mode == 'weighted':
        degree = data.sum(axis = -1)
    if mode == 'binary':
        bdata = data.copy()
        bdata[data > 0] = 1
        degree = bdata.sum(axis = -1)
    return degree
def rich_cl(Y):
    rich_club = []
    for one in Y:
        G = nx.from_numpy_array(one)
        rcv = nx.rich_club_coefficient(G, normalized=False)
        a = set(rcv.keys())
        b = set(range(68))
        add_keys = b.difference(a)
        for idx in add_keys:
            rcv[idx] = 0 
        rich_club += [np.array(list(rcv.values()))]
    return np.array(rich_club)
def av_clus(Y):
    av_clus = []
    for one in Y:
        G = nx.from_numpy_array(one)
        av_clus += [nx.average_clustering(G, weight='weight')]
    return np.array(av_clus)
def mod_score(Y):
    mod_score = []
    for one in Y:
        G = nx.from_numpy_array(one)
        part = community.best_partition(G)
        mod_score += [community.modularity(part, G, weight='weight')]
    return np.array(mod_score)
def local_measure(Y, name):
    measure = []
    for one in Y:
        G = nx.from_numpy_array(one)
        if name == 'degree_centrality':
            measure+= [np.array(list(getattr(nx, name)(G).values()))]
        
        elif name == 'closeness_centrality':
            measure+= [np.array(list(getattr(nx, name)(G, distance = 'weight').values()))]
        else:
            measure+= [np.array(list(getattr(nx, name)(G, weight = 'weight').values()))]
        
    return np.array(measure)
def counter_mean_area_thinkness_and_size_of_roi(thinkness, unique_labels):
    '''
    by labels merge and sum -> 68
    
    '''
    un_mean_think = []
    un_num_roi = []
    for l, one in enumerate(thinkness):
        un_label = np.array(list(unique_labels[l][0]) + list(unique_labels[l][1]))
        
        counter = Counter()
        counter.update(un_label)
        t = np.array([v for k, v in sorted(counter.items()) if k!= 0])
        
        un_num_roi += [t]

        one_think = np.array(list(one[0]) + list(one[1]))
        one_think = np.where(one_think < 0, 0, one_think)
        df = pd.DataFrame(data = np.concatenate((one_think[:, np.newaxis], un_label[:, np.newaxis]), axis=1))
        mean_area = df.groupby(by=df[1], axis =0,).mean()[1:]

        un_mean_think += [np.array(mean_area).reshape(-1)]
    return np.array(un_mean_think), np.array(un_num_roi)
def count_mean_roi_volume(thinkness, un_log_jac, mean_area, unique_labels):
    '''
    count volumes -> 68
    
    '''
    
    
    mean_roi_volume = []
    prod = np.multiply(thinkness, np.multiply(np.exp(un_log_jac), mean_area.reshape(1,2,-1)))
    
    for l, one in enumerate(prod):
        un_label = np.array(list(unique_labels[l][0]) + list(unique_labels[l][1]))
        
        one_think = np.array(list(one[0]) + list(one[1]))
        one_think = np.where(np.array(list(thinkness[l][0]) + list(thinkness[l][0])) < 0, 0, one_think)
        
        df = pd.DataFrame(data = np.concatenate((one_think[:, np.newaxis], un_label[:, np.newaxis]), axis=1))
        mean_vol = df.groupby(by=df[1], axis =0,).mean()[1:]

        mean_roi_volume += [np.array(mean_vol).reshape(-1)]
    
    return np.array(mean_roi_volume)

def outliers_normal(volumes):
    idx_b = np.where(volumes > volumes.mean() + 3*volumes.std())[0]
    idx_l = np.where(volumes < volumes.mean() - 3*volumes.std())[0]
    return np.array(list(idx_l) + list(idx_b))


def clear_connec(connectomes):
    '''
    replace by zero edges with freq < 10% along the sample
    
    '''
    bin_connectomes = connectomes.copy()
    bin_connectomes[bin_connectomes > 0] = 1
    freq_mat = bin_connectomes.mean(axis = 0)
    
    freq_edges = squareform(freq_mat)
    perc_edges = np.percentile(freq_edges, 10)
    
    new_connectomes = []
    for one in connectomes:
        t = one.copy()
        t[freq_mat < perc_edges] = 0.
        new_connectomes += [t]
    
    return np.array(new_connectomes)
def get_nodes_attribute(think, log_jacs, meshes_area, mean_labels, node1, node2 = None):
    '''
    think, log_jac, meshes_atra -- some type of data corresponding to meshes
    mean_labels - mapping of each mesh to node 
    
    node1 - integer from 1 to 70, except 4, 39 (because mean_labels from 0 == The Unknown)
    node2 - integer from 1 to 70, except 4, 39
    
    '''
    X = []
    un_label = np.array(list(mean_labels[0]) + list(mean_labels[1]))
    idx1 = np.where(un_label == node1)[0]
    if node2 != None:
        idx2 = np.where(un_label == node2)[0]
    
    for l, one in enumerate(think):
#         if l%100 == 0:
# #             print(l)
#         one_think = np.array(list(one[0]) + list(one[1]))
#         one_log_jac = np.array(list(log_jacs[l][0]) + list(log_jacs[l][1]))
#         one_mesh_area = np.array(list(meshes_area[l][0]) + list(meshes_area[l][1]))
        one_think = one
        one_log_jac = log_jacs[l]
        one_mesh_area = meshes_area[l]
        
        node_think1 = one_think[idx1]
    
        node_log_jac1 = one_log_jac[idx1]
    
        node_mesh_area1 = one_mesh_area[idx1]
        if node2!= None:
            node_think2 = one_think[idx2]

            node_log_jac2 = one_log_jac[idx2]

            node_mesh_area2 = one_mesh_area[idx2]

            x = np.concatenate([node_think1, node_think2, node_log_jac1, node_log_jac2,
                            node_mesh_area1, node_mesh_area2])
        else:
            x = np.concatenate([node_think1, node_log_jac1,node_mesh_area1])
        X += [x]
        
    return np.array(X)


def get_all_attr(think, log_jac, mesh_area, mean_labels, Y = None):
    idx = np.where(mean_labels!=0)[0]
    X = np.concatenate([think[:,idx], log_jac[:, idx], mesh_area[:, idx]], axis = -1)
    if Y != None:
        new_Y = np.array([squareform(y) for y in Y])
        return X, new_Y
    else:
        return X
def get_meshes_coord_tria(tria, mean_labels, node1, node2 = None, coord = None):
        '''
        tria - (number of triangles of mesh, 3) 
        mean_labels - mapping of each mesh to node 
        node1 - integer from 1 to 70, except 4, 39
        node2 - integer from 1 to 70, except 4, 39

        coord - if needed

        '''
        node_tria = []

        un_label = np.array(list(mean_labels[0]) + list(mean_labels[1]))
        idx1 = np.where(un_label == node1)[0]
        meshes = set(idx1)
        if coord != None:
            node_coord = coord[idx1, :]
        if node2 != None:
            idx2 = np.where(un_label == node2)[0]
            if coord != None:
                node_coord = np.concatenate([node_coord,coord[idx2,:]])
            meshes.update(set(idx2))

        for l, one in enumerate(tria):
            if len(meshes.intersection(set(one))) == 3:
                node_tria += [one]

        if coord != None:
            return np.array(node_tria), node_coord
        else:
            return np.array(node_tria)
    