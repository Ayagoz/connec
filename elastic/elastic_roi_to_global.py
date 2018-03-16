import numpy as np
import pandas as pd
import pickle

import os

from data_load import load_data
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, ShuffleSplit



path = '/cobrain/groups/ml_group/data/HCP/HCP/'
connec , thinkness, log_jac, unique_labels, labels, mean_area_eq, idx_subj_connec, idx_subj_think, idx_subj_logjac = load_data(path)

with open('/home/ayagoz/connec/elastic/modularity.pkl', 'rb') as f:
    modularity = pickle.load(f)
with open('/home/ayagoz/connec/elastic/average_clust.pkl', 'rb') as f:
    clustering = pickle.load(f)

thinkness = thinkness.reshape(789, -1)
log_jac = log_jac.reshape(789, -1)
mean_area_eq = mean_area_eq.reshape(-1)
labels = labels.reshape(-1)

idx = np.array(list(range(1,4)) + list(range(5,39)) + list(range(40, 71)))
roi_think = []
for i in range(68):
    loc_idx = np.where(labels == idx[i])[0]
    roi_think += [np.mean(thinkness[:,loc_idx], axis = -1)]
roi_think = np.array(roi_think).T

roi_area = []
for i in range(68):
    loc_idx = np.where(labels == idx[i])[0]
    roi_area += [np.sum(mean_area_eq[loc_idx]*np.exp(log_jac[:, loc_idx]), axis = -1)]
roi_area = np.array(roi_area).T


roi_vol = []
for i in range(68):
    loc_idx = np.where(labels == idx[i])[0]
    roi_vol += [np.sum(mean_area_eq[loc_idx]*np.exp(log_jac[:, loc_idx]) * thinkness[:,loc_idx], axis = -1)]
    
roi_vol = np.array(roi_vol).T

cv = ShuffleSplit(test_size=0.2)


# elastic_param = {'alpha': [ 1e-5, 1e-3, 0.1, 1.0, 5, 10, 20, 50,], 
#                  'l1_ratio': [1e-9, 1e-5, 1e-3, 0.1, 0.25, 0.4, 0.7, 1],
#                  'max_iter':[1000, 10000]}


elastic_param = {'alpha': [1e-5, 1e-3, 0.1, 1.0, 5, 10], 
                 'l1_ratio': [1e-5, 1e-3, 0.1,  0.4, 1, 5],
                 'max_iter':[10000]}





path_res = '/home/ayagoz/connec/results/'
                   
X = np.concatenate([roi_think, roi_area, roi_vol], axis= -1)



print('Fit ElasticNet for modularity')
elastic = ElasticNet()
StanS = StandardScaler()
MinM = MinMaxScaler()
cv = ShuffleSplit(test_size=0.2)
rg = GridSearchCV(elastic, elastic_param, scoring = 'r2', 
                        n_jobs=-1, cv = 10)

rg.fit(X, modularity)
orig = pd.DataFrame.from_dict(rg.cv_results_)
orig = orig.sort_values(by = 'rank_test_score').iloc[:1]

rg.fit(StanS.fit_transform(X), modularity)
st = pd.DataFrame.from_dict(rg.cv_results_)
st = st.sort_values(by = 'rank_test_score').iloc[:1]

rg.fit(MinM.fit_transform(X), modularity)
mm = pd.DataFrame.from_dict(rg.cv_results_)
mm = mm.sort_values(by = 'rank_test_score').iloc[:1]

name_dir = 'exper_elastic_roi_modularity'
if not os.path.exists(path_res + name_dir):
    os.mkdir(path_res + name_dir)


print(orig)
print(st)
print(mm)
orig.to_csv(path_res + name_dir +'/results_orig')
st.to_csv(path_res + name_dir + '/results_st')
mm.to_csv(path_res + name_dir + '/results_mm')


print('Fit ElasticNet for average clustering')
elastic = ElasticNet()
StanS = StandardScaler()
MinM = MinMaxScaler()
cv = ShuffleSplit(test_size=0.2)
rg = GridSearchCV(elastic, elastic_param, scoring = 'r2', 
                        n_jobs=-1, cv = 10)

rg.fit(X, clustering)
orig = pd.DataFrame.from_dict(rg.cv_results_)
orig = orig.sort_values(by = 'rank_test_score').iloc[:1]

rg.fit(StanS.fit_transform(X), clustering)
st = pd.DataFrame.from_dict(rg.cv_results_)
st = st.sort_values(by = 'rank_test_score').iloc[:1]

rg.fit(MinM.fit_transform(X), clustering)
mm = pd.DataFrame.from_dict(rg.cv_results_)
mm = mm.sort_values(by = 'rank_test_score').iloc[:1]

name_dir = 'exper_elastic_roi_av_clust'
if not os.path.exists(path_res + name_dir):
    os.mkdir(path_res + name_dir)

print(orig)
print(st)
print(mm)
orig.to_csv(path_res + name_dir +'/results_orig')
st.to_csv(path_res + name_dir + '/results_st')
mm.to_csv(path_res + name_dir + '/results_mm')