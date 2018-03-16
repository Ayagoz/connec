import numpy as np
import pandas as pd
import pickle
from scipy.spatial.distance import squareform

import os

from data_load import load_data
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import ShuffleSplit



path = '/cobrain/groups/ml_group/data/HCP/HCP/'
connec , thinkness, log_jac, unique_labels, labels, mean_area_eq, idx_subj_connec, idx_subj_think, idx_subj_logjac = load_data(path)


thinkness = thinkness.reshape(789, -1)
log_jac = log_jac.reshape(789, -1)
mean_area_eq = mean_area_eq.reshape(-1)
labels = labels.reshape(-1)

vol = np.exp(log_jac)* mean_area_eq * thinkness

X = np.concatenate([thinkness, log_jac, vol],axis = -1)
Y = connec[:,0,:,:]
Y = np.array([squareform(one) for one in Y])
print(Y.shape)
print(X.shape)

a = np.percentile(Y.std(axis = 0), 45) 
b = np.percentile(Y.std(axis = 0), 65)

a1 = set(np.where(a<=Y.std(axis = 0))[0]) 
b1 = set(np.where(Y.std(axis = 0) <= b)[0])
c = np.array(list(a1.intersection(b1)))

print(c.shape)

alpha = [0.1, 0.5, 1.0] 
l1_ratio = [1e-3, 0.1,  0.4,  1]

sgdr = SGDRegressor(penalty='elasticnet', tol = 1e-3)
rs = ShuffleSplit(n_splits=3, test_size=0.2, random_state=0)




results = pd.DataFrame(columns=['alpha', 'l1_ratio', 'edge', 'mean_test', 'mean_train'])
row = 0
for a in alpha:
    for l in l1_ratio:
        sgdr = SGDRegressor(penalty='elasticnet', tol = 1e-6)
        sgdr.set_params(**{'alpha':a, 'l1_ratio': l})
        print('PARAMS alpha: {}, l1_ratio: {}'.format(a, l))
        for i in sorted(c):
            print('FOR EDGE NUMBER ' + str(i))
            r2_test, r2_train = [], []
            
            for train_index, test_index in rs.split(X):
                X_train, y_train = X[train_index], Y[train_index, i]
                X_test, y_test = X[test_index], Y[test_index, i]
                sgdr.fit(X_train, y_train)
                r2_train += [sgdr.score(X_train, y_train)]
                r2_test += [sgdr.score(X_test, y_test)]
                print('R2 train: {}, test: {}'.format(r2_train[-1], r2_test[-1]))
            print('MEAN R2 TRAIN: {}, TEST: {}'.format(np.mean(r2_train), np.mean(r2_test)))
    results.loc[row] = [a, l, i, np.mean(r2_test), np.mean(r2_train)]
    row += 1

results.to_csv('/home/ayagoz/connec/results/SGDReg/all_to_edge')