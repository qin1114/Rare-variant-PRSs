import numpy as np


def cal_correlation(data1,data2):
    # This function return the fisher z transformed correlation matrix (p1 * p2)    
    data1 = (data1 - data1.mean(axis=0)) / data1.std(axis=0)
    data2 = (data2 - data2.mean(axis=0)) / data2.std(axis=0)
    r=np.dot(np.transpose(data1),data2) / data1.shape[0]
    return r


def get_res_train(X,Y):
    X= np.hstack((X, np.ones((X.shape[0], 1))))
    beta=np.dot(np.linalg.pinv(X),Y)
    Res=Y-np.dot(X,beta)
    return Res, beta


def nets_zscore(x):
    # x : a nsubject * nfeature numpy matrix

    x_zscore=(x-x.mean(axis=0))
    stds=x.std(axis=0)
    index=stds==0
       
    if sum(index)>0:
        stds[index]=0.1
        print('Warning: '+str(sum(index))+' of the features are all zero or constants')
        print('Normalizing them to all zeros ...')
    
    x_zscore=x_zscore/stds
    
    return x_zscore
