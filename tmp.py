import numpy as np
import pandas as pd
import pickle

import os

from sklearn.preprocessing import StandardScaler, MinMaxScaler


from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.metrics import r2_score, make_scorer


path_data = "/cobrain/groups/ml_group/data/HCP/cleaned_data/"

with open(path_data + "mean_mesh_labels", 'rb') as f:
    mean_labels = pickle.load(f)


with open(path_data + "subjects_log_jac", 'rb') as f:
    log_jac = pickle.load(f)

with open(path_data + "subjects_thinkness", 'rb') as f:
    thinkness = pickle.load(f)

with open(path_data + "subjects_mesh_area", 'rb') as f:
    mesh_area = pickle.load(f)


targets_name = ['clustering', 'rich_club', 'betweenness', 'closeness',  'degree_centrality', 'eigenvector']

targets_data = []
for name in targets_name:
    with open(path_data + name, 'rb') as f:
        targets_data  += [pickle.load(f)]

print(np.array(targets_data).shape)


cv = ShuffleSplit(test_size=0.2)



elastic_param = {'alpha': [1e-10, 1e-7, 1e-5, 1e-3, 0.1, 1.0, 5, 10, 20, 50, 70, 100,], 
                  'l1_ratio': [1e-10, 1e-7, 1e-3, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                'max_iter':[1000,10000]}




path_res = '/home/ayagoz/connec/results/'
                   


thinkness = thinkness.reshape(789,-1)
log_jac = log_jac.reshape(789,-1)
mesh_area = mesh_area.reshape(789,-1)

X = np.concatenate([thinkness, log_jac, mesh_area], axis = -1)






cv = ShuffleSplit(test_size=0.2)

from sklearn.model_selection import cross_val_score

elastic = ElasticNet()


for tr, ts in cv.split(X):
    X_train, X_test = X[tr], X[ts]
    y_train, y_test = targets_data[0][tr], targets_data[0][ts]
    elastic = ElasticNet()
    elastic.fit(X_train, y_train)
    y_pred = elastic.predict(X_test)
    print(r2_score(y_test, y_pred))