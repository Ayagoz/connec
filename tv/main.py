def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import numpy as np
import pandas as pd
import pickle
import os
import sys

import parsimony.functions.nesterov.tv as tv
from parsimony.estimators import LinearRegressionL1L2TV, ElasticNet
import parsimony.algorithms as algorithms

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import squareform

from global_utils import load_data, load_meshes_coor_tria, get_data_ij, convert

def one_edge_tv(coord_ij, tria_ij, data_ij, Y_ij, params):
    
    A_ij = tv.linear_operator_from_mesh(coord_ij, tria_ij)
    
    X_train, X_test, y_train, y_test = train_test_split(data_ij, Y_ij, shuffle = True,random_state = 1, test_size = 0.33)
    
    tv_reg = LinearRegressionL1L2TV(l1 = params['l1'], l2 = params['l2'], tv = params['tv'], A=A_ij)
    
    tv_reg.fit(X_train, y_train)

    y_train_pred = tv_reg.predict(X_train)
    y_test_pred = tv_reg.predict(X_test)

    mse_test = mean_squared_error(y_test, y_test_pred)
    r2_test = r2_score(y_test, y_test_pred_tv)
    print('TRAIN MSE TV: {}, TEST MSE TV: {}'.format(mean_squared_error(y_train, y_train_pred),
                                                    mse_test))

    print('TRAIN R2 TV: {}, TEST R2 TV: {}\n'.format(r2_score(y_train, y_train_pred),
                                                    r2_test))
    
    return r2_test
def main():
    file_name = sys.argv[1]
    path_data = sys.argv[2]
    type_X_data = sys.argv[3]
    type_Y_data = str(sys.argv[4])
    all_meshes = bool(sys.argv[5])
    
    
    with open('file.txt', 'r') as f:
        head = f.readline().strip().split(', ')
        params_num = np.array([float(a) for a in f.readline().strip().strip('[').strip(']').split(', ')])
        params = {head[j]: params_num[j] for j in range(len(params_num))}
    
    flag = False
    if (list(sorted(params.keys())) == sorted(['l1', 'l2', 'tv'])):
        print('I will take all edges and compute tv regression with given params')
        flag = True
    elif (type_Y_data == 'edge'):
        print(params)
        print('I will take edge ({0:.0f}, {1:.0f}) and compute tv reg with given params'.format(params['node i'],
                                                                                             params['node j']))
        
    
        
    path = path_data
    bad_subj = ['168139','361941', '257845', '671855'] 
    data, labels, mean_area = load_data(path, bad_subj_include=bad_subj)
    coord, tria = load_meshes_coor_tria(path)
    if type_Y_data == 'edge':
        Y = []
        for one in data.connectome:
            Y += [squareform(one)]
        Y = np.array(Y)
    else:
        if os.path.isfile(type_Y_data):
            with open(type_Y_data, 'rb') as f:
                Y = pickle.load(f)
    print('Y shape: ', Y.shape)
    best = 0.
    with open(file_name, 'w') as f:
        f.write(type_X_data)
        f.write(' ' + type_Y_data)
        f.write('\n')

        for a in list(params.keys()):
            f.write(str(a))
            if a != list(params.keys())[-1]:
                f.write(', ')
        f.write('\n')
        f.write(str(list(params.values())))
        f.write('\n')
        if flag and (type_Y_data == 'edge'):
            for i in range(Y.shape[-1]):
                for j in range(i+1, Y.shape[-1]):
                    if all_meshes:
                        coord_ij = coord
                        tria_ij = tria
                        data_ij = getattr(data, type_X_data)
                    else:
                        coord_ij, tria_ij, data_ij  = get_data_ij(i, j, labels, coord, 
                                                                  tria, convert(getattr(data, type_X_data)))
                    
                    
                    res = one_edge_tv(coord_ij, tria_ij, data_ij, Y[:, i, j], params)
                    
                    f.write('edge ({0:.0f}, {1:.0f}) test r2 score {2:.3f}'.format(i, j, res))
                    f.write('\n')
                    if res > best:
                        print('NEW BEST RESULT R2 {0:.3f} for ({1:.0f}, {2:.0f})'.format(res, i, j))
                        best = res
        else:
            if len(params) > 3:
                i = int(params['node i'])
                j = int(params['node j'])
                params.pop('node i')
                params.pop('node j')
            if all_meshes:
                coord_ij = coord
                tria_ij = tria
                data_ij = getattr(data, type_X_data)
            else:
                coord_ij, tria_ij, data_ij  = get_data_ij(i, j, labels,
                                                          coord, tria,convert(getattr(data, type_X_data)))

            
            if type_Y_data == 'edge':
                y = Y[:, i,j]
            else:
                y = Y
                print('here!')
            res = one_edge_tv(coord_ij, tria_ij, data_ij, y, params)
            
            if type_Y_data == 'edge':
                f.write('edge ({0:.0f}, {1:.0f}) test r2 score {2:.3f}'.format(i, j, res))
                f.write('\n')
                print('RESULT R2 {0:.3f} for ({1:.0f}, {2:.0f})'.format(res, i, j))
            else:
                f.write(type_Y_data + ' test r2 score {0:.3f}'.format(res))
                f.write('\n')
                print(type_Y_data + ' RESULT R2 {0:.3f}'.format(res))

main()
print('finished')