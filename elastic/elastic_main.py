import numpy as np
import pandas as pd 
import os
import sys
import pickle

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor, ElasticNet

from utils import load_data, roi_transformer, meshes_transformer, edge_convert, node_convert 
def meshes_loop(X, y, params):
    alpha = params['alpha']
    l1_ratio = params['l1_ratio']
    results = pd.DataFrame(columns=['alpha', 'l1_ratio', 'y', 'r2_test', 'r2_train'])
    row = 0
    for a in alpha:
        for l in l1_ratio:
            sgdr = SGDRegressor(penalty='elasticnet', tol=1e-6, max_iter=params['max_iter'])
            sgdr.set_params(**{'alpha':a, 'l1_ratio': l})
            print('PARAMS alpha: {}, l1_ratio: {}'.format(a, l))
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
            sgdr.fit(X_train, y_train)
            r2_test = r2_score(y_test, sgdr.predict(X_test))
            r2_train = r2_score(y_train, sgdr.predict(X_train))
            results.loc[row] = [a, l, params['y'], r2_test, r2_train]
            row += 1

    results.to_csv(params['name'])

def roi_loop(X, y, params):
    alpha = params['alpha']
    l1_ratio = params['l1_ratio']
    results = pd.DataFrame(columns=['alpha', 'l1_ratio', 'edge', 'r2_test', 'r2_train'])
    row = 0
    for a in alpha:
        for l in l1_ratio:
            elastic = ElasticNet(normalize=True, max_iter=params['max_iter'])
            elastic.set_params(**{'alpha':a, 'l1_ratio': l})
            print('PARAMS alpha: {}, l1_ratio: {}'.format(a, l))
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
            elastic.fit(X_train, y_train)
            r2_test = r2_score(y_test, elastic.predict(X_test))
            r2_train = r2_score(y_train, elastic.predict(X_train))
            results.loc[row] = [a, l, params['y'], r2_test, r2_train]
            row += 1

    results.to_csv(params['name'])

def main():
    path_data = sys.argv[1]
    type_X_data = sys.argv[2]
    type_Y_data = sys.argv[3]
    multitarget = bool(sys.argv[4])
    
    
    data, labels, mean_area = load_data(path_data)

    if type_X_data == 'all_meshes':
        X = meshes_transformer(data, labels, mean_area)
    
    if type_X_data == 'roi':
        X = roi_transformer(data, labels, mean_area)

    if type_Y_data == 'edge':
        Y = edge_convert(data.connectome)

    if type_Y_data == 'node':
        Y = node_convert(data.connectome)
    flag = False
    if '.pkl' in type_Y_data:
        with open(type_Y_data, 'rb') as f:
            Y = pickle.load(f)
        flag = True
    print(Y.shape)
        
    
    params = {'alpha': [1e-5, 1e-3, 0.1, 1.0, 5, 10, 20,], 
             'l1_ratio': [1e-5, 1e-3, 0.1, 0.25, 0.4, 0.7, 1],
              'max_iter': 10000,
              'y': type_Y_data,
             'name': type_X_data + '_' + type_Y_data}
    if len(Y.shape) == 2:
        if Y.shape[-1] == 68:
            name = 'node'
        if Y.shape[-1] == 2278:
            name = 'edge'

    if type_X_data == 'all_meshes':
        if len(Y.shape) == 1:
            meshes_loop(X, Y, params)
        if len(Y.shape) == 2:
            for i in range(Y.shape[-1]):
                params['y'] = name + '_' + str(i)
                params['name'] = type_X_data + '_' + type_Y_data + '_' + str(i)
                meshes_loop(X, Y[:, i], params)
    if type_X_data == 'roi':
        if len(Y.shape) == 1:
            roi_loop(X, Y, params)      
        else:
            if multitarget == True:
                params['name'] = type_X_data + '_' + type_Y_data + '_multi_target'
                roi_loop(X, Y, params)
            else:
                for i in range(Y.shape[-1]):
                    params['y'] = name + '_' + str(i)
                    params['name'] = type_X_data + '_' + type_Y_data + '_' + str(i)
                    roi_loop(X, Y[:, i], params)

    print('finished')
main()