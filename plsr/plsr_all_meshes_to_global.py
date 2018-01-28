import numpy as np
import pandas as pd
import pickle

import os

from sklearn.preprocessing import StandardScaler, MinMaxScaler


from utils import get_all_attr

from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV, ShuffleSplit


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



cv = ShuffleSplit(test_size=0.2)




plsr_param = {'n_components':[2,3,5,9, 10, 20,40, 50, 70, 80, 99, 120],
              'max_iter':[10, 100, 500, 1000, 2000]}




path_res = '/home/ayagoz/connec/results/'
                   
X = get_all_attr(thinkness, log_jac, mesh_area, mean_labels)

del thinkness
del log_jac
del mesh_area
del mean_labels


print('Fit plsr for modularity')
plsr = PLSRegression()
StanS = StandardScaler()
MinM = MinMaxScaler()
cv = ShuffleSplit(test_size=0.2)
rg = GridSearchCV(plsr, plsr_param, scoring = 'r2', 
                                    n_jobs=1, cv = 3)

rg.fit(X, modularity)
orig = pd.DataFrame.from_dict(rg.cv_results_)
orig = orig.sort_values(by = 'rank_test_score').iloc[:1]

print('Fit plsr standatizied for modularity')
rg.fit(StanS.fit_transform(X), modularity)
st = pd.DataFrame.from_dict(rg.cv_results_)
st = st.sort_values(by = 'rank_test_score').iloc[:1]

print('Fit plsr MinMaxed for modularity')
rg.fit(MinM.fit_transform(X), modularity)
mm = pd.DataFrame.from_dict(rg.cv_results_)
mm = mm.sort_values(by = 'rank_test_score').iloc[:1]

name_dir = 'exper_plsr_all_meshes_modularity'
if not os.path.exists(path_res + name_dir):
    os.mkdir(path_res + name_dir)


orig.to_csv(path_res + name_dir +'/results_orig')
st.to_csv(path_res + name_dir + '/results_st')
mm.to_csv(path_res + name_dir + '/results_mm')

del modularity

with open(path_data + 'average_clustering', 'rb') as f:
    av_clust = pickle.load(f)
print('Fit plsr for average clustering')
plsr = PLSRegression()
StanS = StandardScaler()
MinM = MinMaxScaler()
cv = ShuffleSplit(test_size=0.2)
rg = GridSearchCV(plsr, plsr_param, scoring = 'r2', 
                                    n_jobs=1, cv = 3)

rg.fit(X, av_clust)
orig = pd.DataFrame.from_dict(rg.cv_results_)
orig = orig.sort_values(by = 'rank_test_score').iloc[:1]

rg.fit(StanS.fit_transform(X), av_clust)
st = pd.DataFrame.from_dict(rg.cv_results_)
st = st.sort_values(by = 'rank_test_score').iloc[:1]

rg.fit(MinM.fit_transform(X), av_clust)
mm = pd.DataFrame.from_dict(rg.cv_results_)
mm = mm.sort_values(by = 'rank_test_score').iloc[:1]

name_dir = 'exper_plsr_all_meshes_av_clust'
if not os.path.exists(path_res + name_dir):
    os.mkdir(path_res + name_dir)


orig.to_csv(path_res + name_dir +'/results_orig')
st.to_csv(path_res + name_dir + '/results_st')
mm.to_csv(path_res + name_dir + '/results_mm')