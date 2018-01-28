import numpy as np
import pandas as pd
import pickle

import os


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import ElasticNet

from sklearn.model_selection import GridSearchCV


path_data = "/cobrain/groups/ml_group/data/HCP/cleaned_data/"


with open(path_data + "subjects_roi_thinkness", 'rb') as f:
    roi_think = pickle.load(f)

with open(path_data + "subjects_roi_volume", 'rb') as f:
    roi_volume = pickle.load(f)


with open(path_data + 'modularity', 'rb') as f:
    modularity = pickle.load(f)
    
with open(path_data + 'average_clustering', 'rb') as f:
    av_clust = pickle.load(f)
    

X = np.concatenate([roi_think, roi_volume], axis = -1)

elastic_param = {'alpha': [ 1e-5, 1e-3, 0.1, 1.0, 5, 10, 20, 50,], 
                 'l1_ratio': [1e-9, 1e-5, 1e-3, 0.1, 0.25, 0.4, 0.7, 1],
                 'max_iter':[1000, 10000]}

feature = PolynomialFeatures(degree=2)
new_X = feature.fit_transform(X)

print('Fit ElasticNet for modularity')
elastic = ElasticNet()

rg = GridSearchCV(elastic, elastic_param, scoring = 'r2', 
                        n_jobs=5, cv = 3)

rg.fit(new_X, modularity)
orig1 = pd.DataFrame.from_dict(rg.cv_results_)
orig1 = orig1.sort_values(by = 'rank_test_score').iloc[:1]
print('result for modularity', float(orig1.mean_test_score))

print('Fit ElasticNet for av clust')
elastic = ElasticNet()


rg = GridSearchCV(elastic, elastic_param, scoring = 'r2', 
                        n_jobs=5, cv = 3)

rg.fit(new_X, av_clust)
orig2 = pd.DataFrame.from_dict(rg.cv_results_)
orig2 = orig2.sort_values(by = 'rank_test_score').iloc[:1]
print('result for av _clust', float(orig2.mean_test_score))
path_res = '/home/ayagoz/connec/results/'
if not os.path.exists(path_res + 'non_lin_roi'):
    os.mkdir(path_res + 'non_lin_roi')
new_path = path_res + 'non_lin_roi/'
orig1.to_csv(new_path+'modularity')
orig2.to_csv(new_path + 'av_clust')