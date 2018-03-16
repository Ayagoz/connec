import numpy as np
import pandas as pd
import pickle

import os

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from data_load import load_data, load_meshes_coor_tria
from tv_utils import get_data_ij


from scipy.spatial.distance import squareform

from sklearn.model_selection import ShuffleSplit
from sklearn.cross_decomposition import PLSRegression


path = '/cobrain/groups/ml_group/data/HCP/HCP/'
connec , thinkness, log_jac, unique_labels, labels, mean_area_eq, idx_subj_connec, idx_subj_think, idx_subj_logjac = load_data(path)

# path_tria= "/cobrain/groups/ml_group/data/HCP/HCP/"
path_new_tria = '/cobrain/groups/ml_group/data/HCP/HCP/'

name_simp = '_200_mean_IC5.m'
# name_orig = '_200_mean.m' 

# orig_coord, orig_tria = load_meshes_coor_tria(path_tria, name_orig)
new_coord , new_tria = load_meshes_coor_tria(path_new_tria, name_simp)


cv = ShuffleSplit(n_splits = 3, test_size=0.2)


path_res = '/home/ayagoz/connec/results/'
                   
Y = connec[:,0,:,:]
Y = np.array([squareform(one) for one in Y])
print(Y.shape)

m = int(len(new_coord)/2)
new_log_jac = np.concatenate([log_jac[:,0, :m], log_jac[:,1,:m]], axis = 1)

new_thick = np.concatenate([thinkness[:,0, :m], thinkness[:,1,:m]], axis = 1)

new_area = np.concatenate([mean_area_eq[0, :m] * np.exp(new_log_jac[:,:m]), 
                           mean_area_eq[1, :m] * np.exp(new_log_jac[:,m:])], axis = 1)

new_labels = np.concatenate([labels[0,:m], labels[1,:m]], axis = 0)

print('new ', new_log_jac.shape, new_thick.shape, new_area.shape, new_labels.shape)

X = np.concatenate([new_thick, new_log_jac, new_area], axis = 1)
X.shape, Y.shape

row = 0
results = pd.DataFrame(columns=['n_components', 'max_iter',  'mean_test', 'mean_train'])

components= [2, 5, 10, 50, 80, 99, 130]
max_iter = [500, 1000, 2000]

for n in components:
    for m in max_iter:
        r2_train = []
        r2_test = []
        plsr = PLSRegression(n_components=n, max_iter=m)
        print('PARAMS n_components: {}, max_iter: {}'.format(n, m))
        for train_index, test_index in cv.split(X):
            X_train, y_train = X[train_index], Y[train_index]
            X_test, y_test = X[test_index], Y[test_index]
            plsr.fit(X_train, y_train)
            r2_train += [plsr.score(X_train, y_train)]
            r2_test += [plsr.score(X_test, y_test)]
            print('R2 train: {}, test: {}'.format(r2_train[-1], r2_test[-1]))
        print('MEAN R2 TRAIN: {}, TEST: {}'.format(np.mean(r2_train), np.mean(r2_test)))
    results.loc[row] = [n, m, np.mean(r2_test), np.mean(r2_train)]
    row += 1



results.to_csv('/home/ayagoz/connec/results/PLSReg/all_ICO5_to_edge')        