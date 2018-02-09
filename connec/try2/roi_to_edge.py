import numpy as np
import pandas as pd
import pickle
import os 

from utils import mean_roi_thick, area, vol

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from joblib import Parallel, delayed



path_data = "/nmnt/x01-hdd/HCP/data/"

with open(path_data + "not_normed_connectomes", 'rb') as f:
    Y = pickle.load(f)

with open(path_data + "normed_connectomes", 'rb') as f:
    norm_Y = pickle.load(f)


with open(path_data + "subjects_roi_thinkness", 'rb') as f:
    roi_think = pickle.load(f)
    
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

new_roi_thick = mean_roi_thick(thinkness, mean_mesh_labels)

new_roi_area = area(mean_mesh_labels, log_jac, mesh_area)

new_roi_vol = vol(mean_mesh_labels, log_jac, mesh_area, thinkness)

print('finished data load')
def main(data, Y):

    params = {'alpha': np.linspace(3*1e-8, 1, 12).tolist() + np.linspace(1.1, 10, 3).tolist(),
              'l1_ratio': np.linspace(0, 1, 11), 
              'max_iter': [100, 1000, 10000]}
    
    gr = GridSearchCV(ElasticNet(), params, scoring='r2', n_jobs = 3, cv = ShuffleSplit(n_splits=3))
    
    X = PolynomialFeatures(degree = 2).fit_transform(data)
    
    gr.fit(X, Y)
    
    res = pd.DataFrame.from_dict(gr.cv_results_)
    
    res = res.sort_values(by = 'rank_test_score').iloc[:1]
    print('best result :{0:.3f}'.format(np.array(res.mean_test_score)[0]))
    return np.array(res.mean_test_score)

mask = Y.sum(axis = 0).copy()
mask[mask > 0] = 1
tuples = []
for i in range(68):
    for j in range(i+1, 68):
        if mask[i,j] == 1:
            tuples += [(i,j)]

print(len(tuples))
norm_new_roi_thick = np.array([new_roi_thick[:,i]/ new_roi_thick.mean(axis = -1) for i in range(68)]).T
norm_new_roi_area = np.array([new_roi_area[:,i]/ new_roi_area.sum(axis = -1) for i in range(68)]).T
norm_new_roi_vol = np.array([new_roi_vol[:,i]/ new_roi_vol.sum(axis = -1) for i in range(68)]).T

print(norm_new_roi_area.shape, norm_new_roi_thick.shape, norm_new_roi_vol.shape)


            
all_res = Parallel(n_jobs=5)(delayed(main)(new_roi_thick[:,[one[0],one[1]]], Y[:,one[0],one[1]]) for one in tuples)


with open('res/result_elastic_roi_thick_to_edge', 'wb') as f:
    pickle.dump(np.array(all_res), f)

print('finished thick')
all_res = Parallel(n_jobs=5)(delayed(main)(new_roi_area[:,[one[0],one[1]]], Y[:,one[0],one[1]]) for one in tuples)


with open('res/result_elastic_roi_area_to_edge', 'wb') as f:
    pickle.dump(np.array(all_res), f)


print('finished area')

all_res = Parallel(n_jobs=5)(delayed(main)(new_roi_vol[:,[one[0],one[1]]], Y[:,one[0],one[1]]) for one in tuples)


with open('res/result_elastic_roi_vol_to_edge', 'wb') as f:
    pickle.dump(np.array(all_res), f)


print('finished vol')
                      
                      

norm_all_res = Parallel(n_jobs=5)(delayed(main)(norm_new_roi_thick[:,[one[0],one[1]]], norm_Y[:,one[0],one[1]]) for one in tuples)

with open('res/result_elastic_norm_roi_thick_to_edge', 'wb') as f:
    pickle.dump(np.array(norm_all_res), f)



print('finished normed thick')

norm_all_res = Parallel(n_jobs=5)(delayed(main)(norm_new_roi_area[:,[one[0],one[1]]], norm_Y[:,one[0],one[1]]) for one in tuples)

with open('res/result_elastic_norm_roi_area_to_edge', 'wb') as f:
    pickle.dump(np.array(norm_all_res), f)



print('finished normed area')
norm_all_res = Parallel(n_jobs=5)(delayed(main)(norm_new_roi_vol[:,[one[0],one[1]]], norm_Y[:,one[0],one[1]]) for one in tuples)

with open('res/result_elastic_norm_roi_vol_to_edge', 'wb') as f:
    pickle.dump(np.array(norm_all_res), f)


print('finished vol')    

    
print('I have finished this shit')
