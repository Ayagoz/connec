import numpy as np
import pandas as pd 
import os
import sys
import pickle

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression, PLSCanonical, PLSSVD

from utils import load_data, roi_transformer, meshes_transformer, edge_convert, node_convert 

def PLSReg_loop(X, y, params, number_rnd):
        
    row = 0
    results = pd.DataFrame(columns=['n_components', 'max_iter', 'random_state', 'r2_test', 'r2_train'])
    
    n_components = params['n_components']
    max_iter = params['max_iter']
    random_states = np.random.randint(0, 10000, number_rnd)
    
    x_load = []
    y_load = []
    
    for n in n_components:
        for m in max_iter:
            x_loc = []
            y_loc = []
            for r in random_states:
                print('PARAMS n_components: {}, max_iter: {}, random_state: {}'.format(n, m, r))
                plsr = PLSRegression()
                plsr.set_params(**{'n_components':n, 'max_iter':m})
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=r)
                #plsr.fit(X_train, y_train)

                #x_loc += [plsr.x_loadings_]
                #y_loc += [plsr.y_loadings_]
                
                r2_test = 0.#plsr.score(X_test, y_test)
                r2_train = 0.#plsr.score(X_train, y_train)                

                results.loc[row] = [n, m, r, r2_test, r2_train]
                row += 1
                
            x_load += [np.array(x_loc)]
            y_load += [np.array(y_loc)]

    x_load = np.array(x_load)
    y_load = np.array(y_load)
    
    
    results.to_csv(params['name'] + '_results')
    
    with open(params['name'] + '_x_load', 'wb') as f:
        pickle.dump(x_load, f)
    
    with open(params['name'] + '_y_load', 'wb') as f:
        pickle.dump(y_load, f)

def main():
    path_data = sys.argv[1]
    type_X_data = sys.argv[2]
    morphometry = sys.argv[3]
    type_Y_data = sys.argv[4]
    number_rnd = int(sys.argv[5])
    
    
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
        
    
    params = {'n_components': [1, 2, 3, 5, 10, 20, 50, 90, 120], 
              'max_iter': [500, 1000, 2000],
              'y': type_Y_data,
              'name': type_X_data  + '_' + morphometry + '_' + type_Y_data}
    
    if morphometry != 'all':
        X = getattr(X, morphometry)
    PLSReg_loop(X, Y, params, number_rnd)
    print('finished')
main()