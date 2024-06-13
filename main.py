import argparse
import os, sys, time

import torch 
import torch.nn as nn
import torch.optim as optim

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score,accuracy_score

from Data_Processing import load_benchmark
from run_retrain_TabSRALinear import run_experiment_batch_TabSRALinear
from run_retrain_EBM import run_experiment_batch_EBM
from run_ARF import run_experiment_batch_ARF
from run_LR import run_experiment_batch_LR
from run_LB import run_experiment_batch_LB
from run_LB_HT import run_experiment_batch_LB_HT
from run_HAT import run_experiment_batch_HAT
from run_HT import run_experiment_batch_HT
from run_retrain_XGBoost import run_experiment_batch_XGBoost
from run_B_XGBoost import run_experiment_batch_B_XGBoost
from run_B_EBM import run_experiment_batch_B_EBM
from run_Update_TabSRALinear import run_experiment_batch_Upadate_TabSRALinear
from run_DT import run_experiment_batch_DT
def parse_args():
    parser = argparse.ArgumentParser()
    
    ## Dataset and split
    parser.add_argument("--dataset_name",type=str,
                        default='agr_a',
                        choices=['agr_a','agr_g','hyper_f','sea_a','sea_g'],
                        help='The name of the benchmark dataset'
                        )
    parser.add_argument("--data_path",type=str,
                        default='Datasets/',
                        help='Data path'
                        )

    parser.add_argument("--seed",type=int,
                        default=42,
                        help='Seed for reproductibility'
                        )
    parser.add_argument("--init_fit_ratio",type=float,
                        default=0.2
                        )
    ## Model
    ### LB==>LB_LR
    ### TabSRALinear==>TabSRA
    parser.add_argument("--model_name",type=str,
                        default='retrain_TabSRALinear',
                        choices=['retrain_TabSRALinear','retrain_EBM', 'ARF', 'LR','LB', 'LB_HT', 'HAT', 'HT', 'retrain_XGBoost','B_XGBoost','B_EBM','Update_TabSRALinear','DT'],
                        help='The name of the benchmark model'
                        )   

 
    ## Use mode
    parser.add_argument("--n_delays",nargs='+',type=int,
                        help='the average length or of delays'
                        )
    parser.add_argument("--n_windows",type=int,
                        default=10000,
                        help='average size of each batch'
                        )
    parser.add_argument("--static_optim_ntrial",type=int,
                        default=30,
                        help='number of tuning iterations for the static pretraining'
                        )
    

    
    config_options = parser.parse_args()
    return config_options


def main():
    config_options = parse_args()
    ## load datasets
    dataset_csv, features,target = load_benchmark(name=config_options.dataset_name, data_path=config_options.data_path)   
    
    if config_options.model_name=='retrain_TabSRALinear':
        run_times, results_,results_accuracy = run_experiment_batch_TabSRALinear(
                                  model_name= config_options.model_name,
                                  dataset= dataset_csv,
                                  dataset_name= config_options.dataset_name,
                                  n_delays= config_options.n_delays,
                                  features= features,
                                  target= target,
                                  metric= [roc_auc_score, accuracy_score],
                                  n_windows = config_options.n_windows,
                                  static_optim_ntrial= config_options.static_optim_ntrial,
                                  seed= config_options.seed,
                                  moment=0,
                                  init_fit_ratio = config_options.init_fit_ratio,
                                  delay='delay',
                                  batch_index='batch_index'  
    )
        
    elif  config_options.model_name=='retrain_EBM':
        run_times, results_,results_accuracy = run_experiment_batch_EBM(
                                          model_name=config_options.model_name,
                                          dataset=dataset_csv,
                                          dataset_name=config_options.dataset_name,
                                          n_delays=config_options.n_delays,
                                          features=features,
                                          target=target,
                                          metric=[roc_auc_score, accuracy_score],
                                          n_windows = config_options.n_windows,
                                          static_optim_ntrial= config_options.static_optim_ntrial,
                                          seed= config_options.seed,
                                          moment=0,
                                          init_fit_ratio = config_options.init_fit_ratio,
                                          delay='delay',
                                          batch_index='batch_index'  
        )
    elif  config_options.model_name=='ARF':
        run_times, results_,results_accuracy = run_experiment_batch_ARF(
                                          model_name=config_options.model_name,
                                          dataset=dataset_csv,
                                          dataset_name=config_options.dataset_name,
                                          n_delays=config_options.n_delays,
                                          features=features,
                                          target=target,
                                          metric=[roc_auc_score, accuracy_score],
                                          n_windows = config_options.n_windows,
                                          static_optim_ntrial=config_options.static_optim_ntrial,
                                          seed=config_options.seed,
                                          moment=0,
                                          init_fit_ratio = config_options.init_fit_ratio,
                                          delay='delay',
                                          batch_index='batch_index'  
        )
    elif  config_options.model_name=='LR':
        run_times, results_,results_accuracy = run_experiment_batch_LR(
                                          model_name=config_options.model_name,
                                          dataset=dataset_csv,
                                          dataset_name=config_options.dataset_name,
                                          n_delays=config_options.n_delays,
                                          features=features,
                                          target=target,
                                          metric=[roc_auc_score, accuracy_score],
                                          n_windows = config_options.n_windows,
                                          static_optim_ntrial=config_options.static_optim_ntrial,
                                          seed=config_options.seed,
                                          moment=0,
                                          init_fit_ratio = config_options.init_fit_ratio,
                                          delay='delay',
                                          batch_index='batch_index'  
        )
    elif  config_options.model_name=='LB':
        run_times, results_,results_accuracy = run_experiment_batch_LB(
                                          model_name=config_options.model_name,     
                                          dataset=dataset_csv,  
                                          dataset_name=config_options.dataset_name,
                                          n_delays=config_options.n_delays,
                                          features=features,
                                          target=target,
                                          metric=[roc_auc_score, accuracy_score],
                                          n_windows = config_options.n_windows,
                                          static_optim_ntrial=config_options.static_optim_ntrial,
                                          seed=config_options.seed,
                                          moment=0, 
                                          init_fit_ratio = config_options.init_fit_ratio,   
                                          delay='delay',
                                          batch_index='batch_index'  
        ) 
    elif  config_options.model_name=='LB_HT':
        run_times, results_,results_accuracy = run_experiment_batch_LB_HT(
                                          model_name=config_options.model_name,
                                          dataset=dataset_csv,
                                          dataset_name=config_options.dataset_name,
                                          n_delays=config_options.n_delays,
                                          features=features,
                                          target=target,
                                          metric=[roc_auc_score, accuracy_score],
                                          n_windows = config_options.n_windows,
                                          static_optim_ntrial=config_options.static_optim_ntrial,
                                          seed=config_options.seed,
                                          moment=0,
                                          init_fit_ratio = config_options.init_fit_ratio,
                                          delay='delay',
                                          batch_index='batch_index'
        ) 
    elif  config_options.model_name=='HAT':
        run_times, results_,results_accuracy = run_experiment_batch_HAT(
                                          model_name=config_options.model_name,
                                          dataset=dataset_csv,
                                          dataset_name=config_options.dataset_name,
                                          n_delays=config_options.n_delays,
                                          features=features,
                                          target=target,
                                          metric=[roc_auc_score, accuracy_score],
                                          n_windows = config_options.n_windows,
                                          static_optim_ntrial=config_options.static_optim_ntrial,
                                          seed=config_options.seed,
                                          moment=0,
                                          init_fit_ratio = config_options.init_fit_ratio,
                                          delay='delay',
                                          batch_index='batch_index'  
        ) 
    elif  config_options.model_name=='HT':
        run_times, results_,results_accuracy = run_experiment_batch_HT(
                                          model_name=config_options.model_name, 
                                          dataset=dataset_csv,
                                          dataset_name=config_options.dataset_name,
                                          n_delays=config_options.n_delays,
                                          features=features,
                                          target=target,
                                          metric=[roc_auc_score, accuracy_score],
                                          n_windows = config_options.n_windows,
                                          static_optim_ntrial=config_options.static_optim_ntrial,
                                          seed=config_options.seed,
                                          moment=0,
                                          init_fit_ratio = config_options.init_fit_ratio,
                                          delay='delay',
                                          batch_index='batch_index'  
        ) 
    elif  config_options.model_name=='retrain_XGBoost':
        run_times, results_,results_accuracy = run_experiment_batch_XGBoost(
                                          model_name=config_options.model_name,
                                          dataset=dataset_csv,
                                          dataset_name=config_options.dataset_name,
                                          n_delays=config_options.n_delays,
                                          features=features,
                                          target=target,
                                          metric=[roc_auc_score, accuracy_score],
                                          n_windows = config_options.n_windows,
                                          static_optim_ntrial=config_options.static_optim_ntrial,
                                          seed=config_options.seed,
                                          moment=0,
                                          init_fit_ratio = config_options.init_fit_ratio,
                                          delay='delay',
                                          batch_index='batch_index'  
        ) 
    elif  config_options.model_name=='B_XGBoost':
        run_times, results_,results_accuracy = run_experiment_batch_B_XGBoost(
                                          model_name=config_options.model_name, 
                                          dataset=dataset_csv,
                                          dataset_name=config_options.dataset_name,
                                          n_delays=config_options.n_delays,
                                          features=features,
                                          target=target,
                                          metric=[roc_auc_score, accuracy_score],
                                          n_windows = config_options.n_windows,
                                          static_optim_ntrial=config_options.static_optim_ntrial,
                                          seed=config_options.seed,
                                          moment=0,
                                          init_fit_ratio = config_options.init_fit_ratio,
                                          delay='delay',
                                          batch_index='batch_index'  
        ) 
    elif  config_options.model_name=='B_EBM':
        run_times, results_,results_accuracy = run_experiment_batch_B_EBM(
                                          model_name=config_options.model_name,
                                          dataset=dataset_csv,
                                          dataset_name=config_options.dataset_name,
                                          n_delays=config_options.n_delays,
                                          features=features,
                                          target=target,
                                          metric=[roc_auc_score, accuracy_score],
                                          n_windows = config_options.n_windows,
                                          static_optim_ntrial=config_options.static_optim_ntrial,
                                          seed=config_options.seed,
                                          moment=0,
                                          init_fit_ratio = config_options.init_fit_ratio,
                                          delay='delay',
                                          batch_index='batch_index'  
        ) 

    elif  config_options.model_name=='Update_TabSRALinear':
        run_times, results_,results_accuracy = run_experiment_batch_Upadate_TabSRALinear(
                                          model_name=config_options.model_name,
                                          dataset=dataset_csv,
                                          dataset_name=config_options.dataset_name,
                                          n_delays=config_options.n_delays,
                                          features=features,
                                          target=target,
                                          metric=[roc_auc_score, accuracy_score],
                                          n_windows = config_options.n_windows,
                                          static_optim_ntrial=config_options.static_optim_ntrial,
                                          seed=config_options.seed,
                                          moment=0,
                                          init_fit_ratio = config_options.init_fit_ratio,
                                          delay='delay',
                                          batch_index='batch_index'  
        ) 
    elif  config_options.model_name=='DT':
        run_times, results_,results_accuracy = run_experiment_batch_DT(
                                          model_name=config_options.model_name,
                                          dataset=dataset_csv,
                                          dataset_name=config_options.dataset_name,
                                          n_delays=config_options.n_delays,
                                          features=features,
                                          target=target,
                                          metric=[roc_auc_score, accuracy_score],
                                          n_windows = config_options.n_windows,
                                          static_optim_ntrial=config_options.static_optim_ntrial,
                                          seed=config_options.seed,
                                          moment=0,
                                          init_fit_ratio = config_options.init_fit_ratio,
                                          delay='delay',
                                          batch_index='batch_index'  
        )  
if __name__=='__main__':
    main()
