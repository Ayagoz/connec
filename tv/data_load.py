import numpy as np
import os

def load_data(path = '/home/bgutman/datasets/HCP/'):
    '''
    
    return: connectomes, thinkness, log_jacobian, unique_labels, labels_think_mean, mean_area, subj_name_idx
    
    connectomes content 793 subj
    
    thinkness content 814 subj
    
    log_jac content 814 subj
    
    
    
    unique_labels content 816 subj
    
    # SUBJECT - (FULL, NORMED) - CONNECTOME 
    # i - (FULL, NORMED) - 84 * 84

    
    # SUBJECT - HEMISPHERE - THINKNESS
    # i - [L, R] - 163842
    
    # SUBJECT - HEMISHPERE - LOG JACOBIAN
    # i - [L, R] - 163842
    
    # SUBJECT - HEMISHPERE - UNIQUE MESHES LABELING 
    # i - [L, R] -163842  
    (unique labels for thinkness into connectome's nodes)
    
    
    # [L, R] - labels for thinkness into connectome's nodes
    # [L, R] - mean area equalized 
    # [L, R] - unique labels for thinkness into connectome's nodes
    
    bad, very very bad subjects ['101309','168139','361941', '257845', '671855'] 
    
    TOTAL NUMBER OF SUBJECT 789
    
    '''
    
    path_connectomes = path + 'Connectomes/' 
    idx = np.array(list(range(3)) + list(range(4,38)) + list(range(39, 70)))
    connectomes = []
    idx_subj_connec = []

    for foldername in sorted(os.listdir(path_connectomes)):
        tmp = []
        if foldername!= '101309' and foldername != '168139' and foldername != '361941' and foldername!= '257845' and foldername!= '671855':
            idx_subj_connec += [foldername]
            for filename in sorted(os.listdir(path_connectomes + foldername + '/con/')):
                    with open(path_connectomes+ foldername + '/con/' + filename, 'rb') as f:
                        t = np.loadtxt(f, dtype='float32')
                        np.fill_diagonal(t,0)
                        tmp += [t[np.ix_(idx,idx)]] 

            connectomes += [tmp]




    path_thickness = path + 'FreeSurfer/thickness/'
    thinkness = []
    idx_subj_think = []

    for foldername in sorted(os.listdir(path_thickness)):
        if foldername in idx_subj_connec:
            idx_subj_think += [foldername]
            tmp = []
            with open(path_thickness + foldername + '/' + 'LH_thick_smooth_2e-4.raw', 'rb') as f:
                    t = np.fromfile(f, count=-1 ,dtype='float32')
                    t = np.where(t < 0, 0, t)
                    tmp += [t]
            with open(path_thickness + foldername + '/' + 'RH_thick_smooth_2e-4.raw', 'rb') as f:
                    t = np.fromfile(f, count=-1 ,dtype='float32')
                    t = np.where(t < 0, 0, t)
                    tmp += [t]
            thinkness += [tmp]





    path_log_jac = path + 'FreeSurfer/LogJacobian/'
    log_jac = []
    idx_subj_logjac = []

    for foldername in sorted(os.listdir(path_log_jac)):
        if foldername in idx_subj_connec:
            idx_subj_logjac += [foldername]
            tmp = []
            with open(path_log_jac + foldername + '/' + 'LH_LogJac_2e-4_eq.raw', 'rb') as f:
                    tmp += [np.fromfile(f, count=-1 ,dtype='float32')]
            with open(path_log_jac + foldername + '/' + 'RH_LogJac_2e-4_eq.raw', 'rb') as f:
                    tmp += [np.fromfile(f, count=-1 ,dtype='float32')]
            log_jac += [tmp]
    
    path_unique_labels = path + 'FreeSurfer/Segmentation/ifs/loni/ccb/CCB_SW_Tools/SurfaceTools/YalinWang_BorisGutman/IITP/HCP/FreeSurfer/Segmentation/'

    unique_labels = []
    idx_subj_unlab = []

    for foldername in sorted(os.listdir(path_unique_labels)):
        if foldername in idx_subj_connec:
            idx_subj_unlab += [foldername]
            tmp = []

            with open(path_unique_labels + foldername + '/' + 'LH_seg_sampled.raw', 'rb') as f:
                tmp += [np.fromfile(f, count=-1 ,dtype='float32')]
            with open(path_unique_labels + foldername + '/' + 'RH_seg_sampled.raw', 'rb') as f:
                tmp += [np.fromfile(f, count=-1 ,dtype='float32')]
            unique_labels += [tmp]

    labels = []    
    with open(path + 'LH_labels_MajVote.raw', 'rb') as f:
        labels += [np.fromfile(f, count=-1 ,dtype='float32')]
    
    with open(path + 'RH_labels_MajVote.raw', 'rb') as f:
        labels += [np.fromfile(f, count=-1 ,dtype='float32')]
    labels = np.array(labels)
    
    
    mean_area_eq = []
    with open(path + 'LH_mean_area_equalized.raw', 'rb') as f:
        mean_area_eq += [np.fromfile(f, count=-1 ,dtype='float32')]
    
    with open(path + 'RH_mean_area_equalized.raw', 'rb') as f:
        mean_area_eq += [np.fromfile(f, count=-1 ,dtype='float32')]
    mean_area_eq = np.array(mean_area_eq)
    
    
    print('Connectomes: ', np.array(connectomes).shape)

    print('Thinkness: ', np.array(thinkness).shape)

    print('Log Jacob: ', np.array(log_jac).shape)
    
    print('Unique Labels: ', np.array(unique_labels).shape)
    
    print('Mean Labels: ', labels.shape)
    
    print('Mean Area eq: ', mean_area_eq.shape)
    
    
    
    return np.array(connectomes), np.array(thinkness), np.array(log_jac), np.array(unique_labels), labels, mean_area_eq, np.array(idx_subj_connec), np.array(idx_subj_think), np.array(idx_subj_logjac)

def load_meshes_coor_tria(path, name):
    '''
    read and load meshes coordinates and triangles 
    return : coord, triangles
    
    #num of mesh - [L, R]
    
    '''
    LH_coord = []
    LH_tria = []
    with open(path + 'LH' + name, 'r') as f:
        f.readline()
        for line in f:
            a = line.strip('\n').split(' ')
            if a[0] == 'Vertex':
                LH_coord += [np.array([float(x) for x in a[2:-1]])]
            if a[0] == 'Face':
                LH_tria += [np.array([int(x)-1 for x in a[2:]])]
    RH_coord = []
    RH_tria = []
    with open(path +'RH' + name, 'r') as f:
        f.readline()
        for line in f:
            a = line.strip('\n').split(' ')
            if a[0] == 'Vertex':
                RH_coord += [np.array([float(x)-1 for x in a[2:-1]])]
            if a[0] == 'Face':
                RH_tria += [np.array([int(x)+len(LH_coord) - 1 for x in a[2:]])]
    coord = np.array(LH_coord + RH_coord)
    tria = np.array(LH_tria + RH_tria)
    
    print('Meshes coordinates shape: ', coord.shape)
    print('Number of triangles of meshes: ', tria.shape)
    return coord, tria
    
    
    
    
    