#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore")
import optuna
import joblib
import time
import datetime as dt
import os, sys
from copy import deepcopy
import numpy as np
import pandas as pd 
from sklearn.metrics import roc_auc_score,accuracy_score,average_precision_score
from sklearn.model_selection import train_test_split
from river import stream
from river import preprocessing
import pickle
import torch
import torch.nn as nn

from torch.optim.lr_scheduler import ReduceLROnPlateau
from skorch.callbacks import EarlyStopping,LRScheduler, TrainEndCheckpoint, EpochScoring
from skorch.dataset import Dataset
from skorch.helper import predefined_split

from Data_Processing import load_benchmark, get_data_split, generate_X_y
from SRAModels import TabSRALinearClassifier,InputShapeSetterTabSRA,reset_seed_

def optimize_static_params_TabSRALinear(dataset_csv,features,target,init_fit_ratio=0.2, n_trial=30, time_out=6*3600,seed=42,name='TabSRALinear_'):
    def objective_TabSRALinear(trial):
        
        params = {
                  'optimizer__lr': trial.suggest_loguniform('optimizer__lr', 1e-3, 1e-1),
                  'optimizer__weight_decay': trial.suggest_loguniform('optimizer__weight_decay', 1e-6, 1e-4),
                  'module__n_hidden_encoder': trial.suggest_int("module__n_hidden_encoder",1, 2, step = 1 ),
                  'module__n_head': trial.suggest_int("module__n_head",1, 2, step = 1 ),
                  'module__dim_head':trial.suggest_int("module__dim_head",4, 8, step = 4 ),
                  'batch_size':trial.suggest_int("batch_size",256, 512, step = 256 ),
                  'module__dropout_rate':trial.suggest_discrete_uniform('module__dropout_rate', 0.0, 0.5, 0.1)
                  }
        static_params = {
                  "module__encoder_bias":True,
                  "module__classifier_bias":True,
                  "random_state":42,
                  "criterion": nn.BCEWithLogitsLoss,
                  "device":'cpu',
                  "verbose":0,
        }
        params.update(static_params)
    
        scoring = EpochScoring(scoring='roc_auc',lower_is_better=False)
        setter = InputShapeSetterTabSRA(regression=False)
        early_stop = EarlyStopping(monitor=scoring.scoring, patience=10,load_best=True,lower_is_better=False, threshold=0.0001,threshold_mode='abs')
        lr_scheduler = LRScheduler(policy=ReduceLROnPlateau, patience=5, min_lr=2e-5,factor=0.2, verbose=1,mode='max', monitor=scoring.scoring)
        call_b = [scoring, setter, early_stop, lr_scheduler]
        model = TabSRALinearClassifier(**params,callbacks=call_b)# 
        
        scaler = preprocessing.StandardScaler()
        
    
        X_train_init,y_train_init, data,stream_features = get_data_split(data=dataset_csv, init_fit_ratio=init_fit_ratio, features=features,target=target)
        X_train_init,X_val_init,y_train_init,y_val_init = train_test_split(X_train_init,y_train_init,test_size=0.3,random_state=seed)
    
        scaler.learn_many(X_train_init)
        X_train_init = scaler.transform_many(X_train_init)
        X_val_init = scaler.transform_many(X_val_init)
        
        valid_dataset = Dataset(X_val_init.values.astype(np.float32),y_val_init.values.astype(np.float32))
        model.train_split = predefined_split(valid_dataset)
        _ = model.fit(X_train_init.values.astype(np.float32),y_train_init.values.astype(np.float32))
    
        y_pred = model.predict_proba(X_val_init.values.astype(np.float32))
        val_result =  roc_auc_score(y_val_init.values, y_pred[:,1]) 
    
        return val_result
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective_TabSRALinear,timeout=time_out,n_trials=n_trial)
    best_trial = study.best_trial
    best_p = best_trial.params
    joblib.dump(best_p,name+'_best_params')

    return best_p
def evaluate_batch_TabSRALinear(params, callbacks, model,scaler, dataset, n_rows, n_delay, n_windows, features, target, metric, seed=42, moment=0, delay='delay', batch_index='batch_index'):

    stream_features = features+['delay','batch_index']
    X,y_target = generate_X_y(dataset=dataset,n_rows=n_rows, n_delay=n_delay,n_windows=n_windows,features=stream_features,target=target,seed=seed)

    dataset_stream = stream.iter_pandas(X,y_target) ## stream data
    batch_index, batch_size = np.unique(X.batch_index,return_counts=True) ## extract id of batch and size 
    windows_count = np.zeros(len(batch_size),dtype=int) ## initialise obtained labels of each batch. 
    Y_pred_buffer = [{} for i in range(len(batch_size))] ## initialise the buffer of predictions 
    Y_true_buffer = [{} for i in range(len(batch_size))] ## initialise the buffer of labels 
    X_buffer = {} ## initialise the buffer of observation, which is removed from the model, once using it for the learning. 
    result_batch= []
    result_batch_accuracy= []
    Len_Batch = []
    ##
    for i, x,y, *kwargs in stream.simulate_qa(dataset_stream, moment=0, delay='delay'):
        
        index_window = int(x.pop('batch_index'))
        delay =  x.pop('delay')
        if y is None:
            X_buffer[i] = x
            xto_pred = scaler.transform_one(x)
            y_pred = model.predict_proba(pd.DataFrame([xto_pred])[features].values.astype(np.float32))
            # add the prediction to its corresponding window
            Y_pred_buffer[index_window][i] = y_pred[0][1]
    
            continue
        if y is not None:
            windows_count[index_window]+=1#increase the number of obtained label for the window index_window
            Y_true_buffer[index_window][i] = y
    
            # if the label buffer of the window index_window is, then  display the metric for full batch
            if windows_count[index_window] == batch_size[index_window]:
                y_true_window = np.array(list(Y_true_buffer[index_window].values()))# ground truth
                y_true_index_window = list(Y_true_buffer[index_window].keys())# get all predictions of the current window the index (i) of true label
                Y_pred_buffer_window = Y_pred_buffer[index_window]## extract all predictions of the current window
                y_pred_window = np.array([Y_pred_buffer_window.get(i) for i in y_true_index_window]) # get all predictions array of the current window
                result =  metric[0](y_true_window,y_pred_window)
                result_batch.append(result)
                result_batch_accuracy.append(metric[1](y_true_window,(y_pred_window>0).astype(int)))
                Len_Batch.append(len(y_pred_window))
                print(f"window={index_window}---Score={result}")
                
                X_train_buffer = pd.DataFrame([X_buffer.pop(i) for i in y_true_index_window])[features]
                y_train_buffer = pd.Series(y_true_window)
                X_train_buffer,X_val_buffer,y_train_buffer,y_val_buffer= train_test_split(X_train_buffer,y_train_buffer,test_size=0.3,random_state=42)
                ###                
                ###
                reset_seed_(seed=seed)
                scaler.learn_many(X_train_buffer)
                X_train_buffer = scaler.transform_many(X_train_buffer)
                X_val_buffer = scaler.transform_many(X_val_buffer)
                new_valid_dataset = Dataset(X_val_buffer.values.astype(np.float32),y_val_buffer.values.astype(np.float32))
                model = TabSRALinearClassifier(**params,callbacks=callbacks)#
                model.train_split = predefined_split(new_valid_dataset)
                try: 
                    _ = model.fit(X_train_buffer.values.astype(np.float32),y_train_buffer.values.astype(np.float32))
                except KeyboardInterrupt:
                    pass 

    return Len_Batch,result_batch,result_batch_accuracy
def run_experiment_batch_TabSRALinear(
                              model_name,
                              dataset,
                              dataset_name,
                              n_delays,
                              features,
                              target,
                              metric,
                              n_windows = 10000,
                              static_optim_ntrial = 25,
                              seed=42,
                              moment=0,
                              init_fit_ratio = 0.3,
                              delay='delay',
                              batch_index='batch_index'
                                     ):

    
    save_config_dir = model_name+'_'+dataset_name
    if os.path.exists(save_config_dir+'best_params'):
        best_params = joblib.load(save_config_dir+'best_params')
    else:    
        best_params = optimize_static_params_TabSRALinear(dataset_csv=dataset,features=features,target=target,n_trial=static_optim_ntrial,init_fit_ratio=init_fit_ratio, time_out=6*3600,seed=42,name=save_config_dir)

    ## split the data for offline tuning and stream evaluation
    X_train_init,y_train_init, data,stream_features = get_data_split(data=dataset, init_fit_ratio=init_fit_ratio, features=features,target=target)

    ## offline optimization
    
    X_train_init,X_val_init,y_train_init,y_val_init = train_test_split(X_train_init,y_train_init,test_size=0.3,random_state=42)
    scaler = preprocessing.StandardScaler()
    scaler.learn_many(X_train_init)
    X_train_init = scaler.transform_many(X_train_init)
    X_val_init = scaler.transform_many(X_val_init)
    
    scoring = EpochScoring(scoring='roc_auc',lower_is_better=False)
    setter = InputShapeSetterTabSRA(regression=False)
    early_stop = EarlyStopping(monitor=scoring.scoring, patience=10,load_best=True,lower_is_better=False, threshold=0.0001,threshold_mode='abs')
    lr_scheduler = LRScheduler(policy=ReduceLROnPlateau, patience=5, min_lr=2e-5,factor=0.2, verbose=1,mode='max', monitor=scoring.scoring)
    callbacks = [scoring, setter, early_stop, lr_scheduler]

    static_params = {
              "module__encoder_bias":True,
              "module__classifier_bias":True,
              "random_state":42,
              "criterion": nn.BCEWithLogitsLoss,
              "device":'cpu',
              "verbose":0,
    }
    
    best_params.update(static_params)
    model = TabSRALinearClassifier(**best_params,callbacks=callbacks)# 
 
    valid_dataset = Dataset(X_val_init.values.astype(np.float32),y_val_init.values.astype(np.float32))
    model.train_split = predefined_split(valid_dataset)
    _ = model.fit(X_train_init.values.astype(np.float32),y_train_init.values.astype(np.float32))    


    model_copies = [deepcopy(model) for m in range(len(n_delays))]
    all_results = []
    all_results_accuracy = []
    run_times = []
    
    df_all_results = pd.DataFrame()
    df_all_results_accuracy = pd.DataFrame()
    
    df_all_results_runtimes = pd.DataFrame()
    df_all_results_runtimes['delays'] = n_delays
    for m, number_delay in enumerate(n_delays):
        start_time = time.time()
        Len_Batch,result_batch,result_batch_accuracy = evaluate_batch_TabSRALinear(params=best_params,
                                                                                   callbacks=callbacks,
                                                                                   model = model_copies[m],
                                                                                   dataset = data,
                                                                                   scaler=scaler,
                                                                                    n_rows=len(data),
                                                                                    n_delay=number_delay,
                                                                                    n_windows=n_windows,
                                                                                    features=features,
                                                                                    target=target,
                                                                                    metric= metric
        )
        if m==0:
            df_all_results["Batch_len"]= Len_Batch
            df_all_results_accuracy["Batch_len"]= Len_Batch

        all_results.append(result_batch)
        all_results_accuracy.append(result_batch_accuracy)

        col_name = model_name+'_delay_'+str(number_delay)
        df_all_results[col_name]=result_batch
        df_all_results_accuracy[col_name]=result_batch_accuracy

        path = dataset_name+model_name+".csv"
        df_all_results.to_csv('aucroc_'+path,index=False)
        df_all_results_accuracy.to_csv('accuracy_'+path,index=False)
        end_time = time.time()
        run_times.append(end_time-start_time)
    df_all_results_runtimes['runtime'] = run_times
    df_all_results_runtimes.to_csv('runtime_'+path,index=False)
    return run_times, all_results, all_results_accuracy               
