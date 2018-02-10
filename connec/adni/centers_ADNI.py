#003_S_0908_1_connect_dil_centerOfGravities
import pandas as pd
import numpy as np
import os 

def load_centers(path, data):
    aid = data['subject_id_file'].tolist()
    flag = 1
    no_centers = []
    #path = 'connectomics/ADNI/ADNI/adni2_centers/'
    centers = pd.DataFrame(columns = ['subject_id', 'scan_id','coordinate'])
    for idx in aid:
        vec = pd.read_csv(path+'/'+idx+'_connect_dil_centerOfGravities.txt')
        vec.drop(vec.index[[3,38]], inplace = True)
        vec.__delitem__('Unnamed: 7')
        vec.__delitem__('region')
        vec.__delitem__('voxel_cordX')
        vec.__delitem__('voxel_cordY')
        vec.__delitem__('voxel_cordZ')
        mat = np.array(vec.as_matrix()).T
        if (mat.all() == 0):
            flag = 0
            no_centers +=[idx]
        #filename = filename[:12]
        subject_id = idx[:-2]
        scan_id = idx[-1:]
        single_subject = pd.DataFrame(data = [[subject_id, scan_id, mat]], 
                                      columns = ['subject_id', 'scan_id','coordinate'])
        if (flag != 0 ):
            if(subject_id != '003_S_5187'):
                centers = centers.append(single_subject)
        flag = 1
    centers.index = centers.subject_id
    no_centers += ['003_S_5187_2']
    no_centers += ['003_S_5187_1']
    print('centers shape : ',centers.shape)
    data.drop(labels=no_centers, inplace=True)
    print('data shape : ', data.shape)
    
    return centers