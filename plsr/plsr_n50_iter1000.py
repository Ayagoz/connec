import numpy as np
import pandas as pd
import pickle

import os
import time
from data_load import load_data, load_meshes_coor_tria
from tv_utils import get_data_ij


from scipy.spatial.distance import squareform

from sklearn.model_selection import ShuffleSplit
from sklearn.cross_decomposition import PLSRegression

from sklearn.datasets import make_regression
path = '/cobrain/groups/ml_group/data/HCP/HCP/'
connec , thinkness, log_jac, unique_labels, labels, mean_area_eq, idx_subj_connec, idx_subj_think, idx_subj_logjac = load_data(path)

path_tria= "/cobrain/groups/ml_group/data/HCP/HCP/"
path_new_tria = '/cobrain/groups/ml_group/data/HCP/HCP/'

name_simp = '_200_mean_IC5.m'
# name_orig = '_200_mean.m' 

# orig_coord, orig_tria = load_meshes_coor_tria(path_tria, name_orig)
new_coord , new_tria = load_meshes_coor_tria(path_new_tria, name_simp)



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
results = pd.DataFrame(columns=['random_state', 'mean_test', 'mean_train'])

# components= [2, 5,]
# max_iter = [2000, 3000]


n = 50
m = 1000
R2_test = []
x_load = []
y_load = []
random_states = np.random.randint(0, 10000, 100)
with open('rnd_states100', 'wb') as f:
    pickle.dump(random_states, f)
st = time.time()
for i in random_states:
    r2_train = []
    r2_test = []
    x_loc = []
    y_loc = []
    cv = ShuffleSplit(n_splits=2, test_size=0.2, random_state=i)
    for train_index, test_index in cv.split(X):
        X_train, y_train = X[train_index], Y[train_index]
        X_test, y_test = X[test_index], Y[test_index]
        plsr = PLSRegression(n_components=n, max_iter=m)
        plsr.fit(X_train, y_train)
        x_loc += [plsr.x_loadings_]
        y_loc += [plsr.y_loadings_]
        r2_train += [plsr.score(X_train, y_train)]
        r2_test += [plsr.score(X_test, y_test)]
        print('R2 train: {}, test: {}'.format(r2_train[-1], r2_test[-1]))
    print('MEAN R2 TRAIN: {}, TEST: {}'.format(np.mean(r2_train), np.mean(r2_test)))
    R2_test += [r2_test]
    x_load += [x_loc]
    y_load += [y_loc]
    results.loc[row] = [i, np.mean(r2_test), np.mean(r2_train)]
    row += 1
end = time.time() - st
print(end)

R2_test = np.array(R2_test)
x_load = np.array(x_load)
y_load = np.array(y_load)

R2_test.shape, x_load.shape, y_load.shape

R2_score = pd.DataFrame(columns=['random_state', 'r2_fold1', 'r2_fold2'])

for i in range(len(random_states)):
    R2_score.loc[i] = [random_states[i], R2_test[i][0], R2_test[i][1]]

R2_score.to_csv('/home/ayagoz/connec/results/R2_test_100rand_state.csv')

x_components = pd.DataFrame(columns=['random_state', 'comp_fold1', 'comp_fold2'])

for i in range(len(random_states)):
    x_components.loc[i] = [random_states[i], x_load[i][0], x_load[i][1]]

y_components = pd.DataFrame(columns=['random_state', 'comp_fold1', 'comp_fold2'])

for i in range(len(random_states)):
    y_components.loc[i] = [random_states[i], y_load[i][0], y_load[i][1]]

x_components.to_csv('/home/ayagoz/connec/results/x_components50_iter1000_rndst100')
y_components.to_csv('/home/ayagoz/connec/results/y_components50_iter1000_rndst100')