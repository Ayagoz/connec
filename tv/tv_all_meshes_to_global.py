import numpy as np
import pandas as pd
import pickle

import os


from utils import get_all_attr

from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV


path_data = "/cobrain/groups/ml_group/data/HCP/cleaned_data/"

with open(path_data + "mean_mesh_labels", 'rb') as f:
    mean_labels = pickle.load(f)

with open(path_data + "subjects_log_jac", 'rb') as f:
    log_jac = pickle.load(f)

with open(path_data + "subjects_thinkness", 'rb') as f:
    thinkness = pickle.load(f)

with open(path_data + "subjects_mesh_area", 'rb') as f:
    mesh_area = pickle.load(f)

with open(path_data + 'modularity', 'rb') as f:
    modularity = pickle.load(f)

    
thinkness = thinkness.reshape(789,-1)
log_jac = log_jac.reshape(789,-1)
mesh_area = mesh_area.reshape(789,-1)






elastic_param = {'alpha': [ 1e-5, 1e-3, 0.1, 1.0, 5, 10, 20, 50,], 
                 'l1_ratio': [1e-9, 1e-5, 1e-3, 0.1, 0.25, 0.4, 0.7, 1],
                 'max_iter':[1000, 10000]}




path_res = '/home/ayagoz/connec/results/'
                   
X = get_all_attr(thinkness, log_jac, mesh_area, mean_labels)

del thinkness
del log_jac
del mesh_area
del mean_labels


print('Fit ElasticNet for modularity')
elastic = ElasticNet()

rg = GridSearchCV(elastic, elastic_param, scoring = 'r2', 
                        n_jobs=1, cv = 3)

rg.fit(X, modularity)
orig = pd.DataFrame.from_dict(rg.cv_results_)
orig = orig.sort_values(by = 'rank_test_score').iloc[:1]

name_dir = 'exper_elastic_all_meshes_modularity'
if not os.path.exists(path_res + name_dir):
    os.mkdir(path_res + name_dir)


orig.to_csv(path_res + name_dir +'/results_orig')


del modularity

with open(path_data + 'average_clustering', 'rb') as f:
    av_clust = pickle.load(f)
print('Fit ElasticNet for average clustering')
elastic = ElasticNet()

rg = GriddSearchCV(elastic, elastic_param, scoring = 'r2', 
                        n_jobs=1, cv = 3)

rg.fit(X, av_clust)
orig = pd.DataFrame.from_dict(rg.cv_results_)
orig = orig.sort_values(by = 'rank_test_score').iloc[:1]


name_dir = 'exper_elastic_all_meshes_av_clust'
if not os.path.exists(path_res + name_dir):
    os.mkdir(path_res + name_dir)


orig.to_csv(path_res + name_dir +'/results_orig')
