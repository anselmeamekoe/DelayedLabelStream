#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np 
import pandas as pd 
def load_benchmark(name ='agr_a', data_path = 'Datasets/'):
    dataset_csv = pd.read_csv(data_path+name+'.csv')
    features = list(dataset_csv.columns)
    target = features.pop(-1)
    return dataset_csv, features,target

def get_data_split(data, features, target, init_fit_ratio=0.3):
    """
    split the  dataset in inital train dataset and the stream data set
    """
    n = len(data)
    data_init = data[:int(init_fit_ratio*n)]
    data = data[int(init_fit_ratio*n):]
    y_train_init = data_init[target]
    X_train_init = data_init[features]
    stream_features = features+['delay','batch_index']
    return X_train_init,y_train_init, data,stream_features    

def generate_X_y(dataset, n_rows, n_delay,n_windows,features, target,seed=42):
    """
    generate stream dataset and add delay and batch information 
    """
    np.random.seed(seed)
    dataset = dataset.iloc[:n_rows]
    dataset['delay']=np.random.poisson(n_delay,dataset.shape[0])
    w_idx = np.random.random_integers(0,(dataset.shape[0]//n_windows)-1,dataset.shape[0])
    w_idx.sort()   
    dataset['batch_index'] = w_idx
    X = dataset[features]
    y_target = dataset.pop(target)
    return X,y_target        