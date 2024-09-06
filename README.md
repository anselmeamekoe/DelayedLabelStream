# Tabular data: label delay and interpretability
 Repository for the paperEvaluating the Efficacy of Instance Incremental vs.Batch Learning in Delayed Label Environments: An Empirical Study on Tabular Data Streaming for Fraud Detection

*In delayed settings, is instance incremental learning the best option regarding predictive performance and computational efficiency?*
### No delay situation: AUCROC Agrawal gradual drift (Agr_g)
<img src="https://github.com/anselmeamekoe/DelayedLabelStream/blob/main/Ressources/Images/AGR_g_DELAY_0.png" width=600px>

### Delay of average 70k instances: AUCROC Agrawal gradual drift (Agr_g)
<img src="https://github.com/anselmeamekoe/DelayedLabelStream/blob/main/Ressources/Images/AGR_g_DELAY_70000.png" width=600px>

## Usage
### Prerequisites
Create a new python environment, install the [requirements](https://github.com/anselmeamekoe/DelayedLabelStream/blob/main/requirements.txt):
``` pip install -r requirements.txt``` 

### Replicating results of the Generated benchmark
1. Clone this repository of your machine
2.  Run the [Notebook](https://github.com/anselmeamekoe/DelayedLabelStream/blob/main/Analysis/notebook_Generated_benchmark.ipynb) for reproducing results
   
NB: To use the notebook, you will need to install it in the python environment you have created using pip for example
   
### Run your own experiments on the Generated benchmark

1. Please download first datasets (of interest) using the link [here](https://github.com/anselmeamekoe/DelayedLabelStream/blob/main/Datasets/DataLink) and place them in **Datasets** folder.
2. For example in  ```python main.py --n_delays 0 1000 70000 --static_optim_ntrial 30 --model_name DT --dataset_name sea_g --init_fit_ratio 0.1 --n_windows 10000``` 
- n_delays: indicates the average label delay, generated following a Poisson distribution of mean  0, 1000, 70000 respectively
- static_optim_ntrial: indicates the number of trial for tuning parameters offline (before the stream evaluation)
- model_name: the model name (e.g., DT is for Decision Tree)
- dataset_name: the dataset name (sea_g here)
- init_fit_ratio: the fraction of the dataset used for the offline optimization  (0.1 in the above example)
- n_windows: average number of instances (following a Poisson distribution) in each evaluation batch (10000 for this example)
  
### Run your own experiments of the Fraud dataset  
For reasons of confidentiality, the fraud dataset is not accessible to the public. 

## Acknowledgments
This work has been done in collaboration between BPCE Group, Laboratoire d'Informatique de Paris Nord (LIPN UMR 7030), DAVID Lab UVSQ-Université Paris Saclay and was supported by the program Convention Industrielle de Formation par la Recherche (CIFRE) of the Association Nationale de la Recherche et de la Technologie (ANRT).
