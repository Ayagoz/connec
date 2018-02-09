import numpy as np
import pandas as pd
import pickle
import os 

from parsimony.algorithms.proximal import FISTA
import parsimony.functions.nesterov.tv as tv
from parsimony.estimators import LinearRegressionL1L2TV
from parsimony.estimators import GridSearchKFoldRegression

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from tv_utils import get_data_ij
from data_load import load_meshes_coor_tria


path_data = "/nmnt/x01-hdd/HCP/data/"
coord, tria = load_meshes_coor_tria(path_data)

with open(path_data + "not_normed_connectomes", 'rb') as f:
    Y = pickle.load(f)
    
with open(path_data + "mean_mesh_labels", 'rb') as f:
    mean_mesh_labels = pickle.load(f)

with open(path_data + "subjects_log_jac", 'rb') as f:
    log_jac = pickle.load(f)

with open(path_data + "subjects_thinkness", 'rb') as f:
    thinkness = pickle.load(f)
    
with open(path_data + "subjects_mesh_area", 'rb') as f:
    mesh_area = pickle.load(f)

log_jac = log_jac.reshape(789, -1)
thinkness = thinkness.reshape(789, -1)
mesh_area = mesh_area.reshape(789, -1)
mean_mesh_labels = mean_mesh_labels.reshape(-1)

grid_params = {'l1': np.linspace(3*1e-10, 0.99, 5), 
          'l2': np.linspace(3*1e-10, 0.99, 5), 
          'tv': np.linspace(3*1e-10, 0.99, 5), 
          'mu': np.linspace(3*1e-10, 0.99, 5)}



mask = Y.sum(axis = 0).copy()
mask[mask > 0] = 1
tuples = []
for i in range(68):
    for j in range(i+1, 68):
        if mask[i,j] == 1:
            tuples += [(i,j)]
def r2_score_tv(est, params, X, y):
    est.set_params(**params)
    
    yhat = est.predict(X)
    
    return r2_score(y, yhat)

a = [[ 4, 26], [38, 60], [4, 38], [26, 38], [4,60], [26, 60]]



for i,j in a:
    
    coord_ij, tria_ij, data_ij  = get_data_ij(i, j, mean_mesh_labels, tria, coord, thinkness)
    X_train, X_test, y_train, y_test = train_test_split(data_ij, Y[:,i,j], test_size = 0.1)
    A_ij = tv.linear_operator_from_mesh(coord_ij, tria_ij)

    tv_reg = LinearRegressionL1L2TV(0.1,0.1,0.1,mu = 0.1, A = A_ij)

    gr = GridSearchKFoldRegression(tv_reg, grid_params, score_function = r2_score_tv, K=3)
    
    gr.fit(X_train, y_train)
    
    yhat = gr.predict(X_test)
    r2_res = r2_score(y_test, yhat)
    
    print(r2_res)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
