import numpy as np
import pandas as pd
import pickle

import os

from sklearn.preprocessing import StandardScaler, MinMaxScaler



from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, ShuffleSplit


path_data = "/cobrain/groups/ml_group/data/HCP/cleaned_data/"


with open(path_data + "subjects_roi_thinkness", 'rb') as f:
    roi_think = pickle.load(f)

with open(path_data + "subjects_roi_volume", 'rb') as f:
    roi_volume = pickle.load(f)


with open(path_data + 'modularity', 'rb') as f:
    modularity = pickle.load(f)
with open(path_data + 'average_clustering', 'rb') as f:
    av_clust = pickle.load(f)
    




cv = ShuffleSplit(test_size=0.2)


elastic_param = {'alpha': [ 1e-5, 1e-3, 0.1, 1.0, 5, 10, 20, 50,], 
                 'l1_ratio': [1e-9, 1e-5, 1e-3, 0.1, 0.25, 0.4, 0.7, 1],
                 'max_iter':[1000, 10000]}





path_res = '/home/ayagoz/connec/results/'
                   
X = np.concatenate([roi_think, roi_volume], axis= -1)



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

rg.fit(X, av_clust)
orig = pd.DataFrame.from_dict(rg.cv_results_)
orig = orig.sort_values(by = 'rank_test_score').iloc[:1]

rg.fit(StanS.fit_transform(X), av_clust)
st = pd.DataFrame.from_dict(rg.cv_results_)
st = st.sort_values(by = 'rank_test_score').iloc[:1]

rg.fit(MinM.fit_transform(X), av_clust)
mm = pd.DataFrame.from_dict(rg.cv_results_)
mm = mm.sort_values(by = 'rank_test_score').iloc[:1]

name_dir = 'exper_elastic_roi_av_clust'
if not os.path.exists(path_res + name_dir):
    os.mkdir(path_res + name_dir)


orig.to_csv(path_res + name_dir +'/results_orig')
st.to_csv(path_res + name_dir + '/results_st')
mm.to_csv(path_res + name_dir + '/results_mm')