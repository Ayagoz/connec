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


targets_name = ['clustering', 'rich_club', 'betweenness', 'closeness',  'degree_centrality', 'eigenvector']

targets_data = []
for name in targets_name:
    with open(path_data + name, 'rb') as f:
        targets_data  += [pickle.load(f)]

print(np.array(targets_data).shape)


cv = ShuffleSplit(test_size=0.2)



plsr_param = {'n_components':[2,3,5,9, 10, 20,40, 50, 70, 80, 99, 120],
              'max_iter':[10, 100, 500, 1000, 2000]}




path_res = '/home/ayagoz/connec/results/'
                   


thinkness = thinkness.reshape(789,-1)
log_jac = log_jac.reshape(789,-1)
mesh_area = mesh_area.reshape(789,-1)

X = get_all_attr(thinkness, log_jac, mesh_area, mean_labels)


for j, target in enumerate(targets_data):
    print(targets_name[j])
    print(target.shape)

    plsr = PLSRegression()
    StanS = StandardScaler()
    MinM = MinMaxScaler()
    cv = ShuffleSplit(test_size=0.2)
    rg = GridSearchCV(plsr, plsr_param, scoring = 'r2', 
                                    n_jobs=1, cv = 3)

    rg.fit(X, target)
    orig = pd.DataFrame.from_dict(rg.cv_results_)
    orig = orig.sort_values(by = 'rank_test_score').iloc[:1]

    rg.fit(StanS.fit_transform(X), target)
    st = pd.DataFrame.from_dict(rg.cv_results_)
    st = st.sort_values(by = 'rank_test_score').iloc[:1]

    rg.fit(MinM.fit_transform(X), target)
    mm = pd.DataFrame.from_dict(rg.cv_results_)
    mm = mm.sort_values(by = 'rank_test_score').iloc[:1]

    name_dir = 'exper_plsr_all_meshes_to_node'
    edge = '/edge[' + targets_name[j] + ']'
    if not os.path.exists(path_res + name_dir):
        os.mkdir(path_res + name_dir)

    if not os.path.exists(path_res + name_dir + edge):
        os.mkdir(path_res + name_dir + edge)
    orig.to_csv(path_res + name_dir + edge +'/results_orig')
    st.to_csv(path_res + name_dir + edge + '/results_st')
    mm.to_csv(path_res + name_dir + edge + '/results_mm')
    

