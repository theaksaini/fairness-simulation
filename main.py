from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# using merged version
from functools import partial
import math

import experimental_setup

def logit(p):
    return math.log(p/(1-p))

def main():
    experimental_conditions = ['Cond1', 'Cond2', 'Cond3', 'Cond4', 'Cond5', 'Cond6']
    param_settings = {}
    param_settings['Cond1'] = { 'groupA_prop':0.20, 'params': [logit(0.10), 0.8, -0.5, -0.3, 0.8],}
    param_settings['Cond2'] = { 'groupA_prop':0.20, 'params': [logit(0.10), 1.0, -0.5, -0.3, 0.8],}
    param_settings['Cond3'] = { 'groupA_prop':0.20, 'params': [logit(0.10), 1.2, -0.5, -0.3, 0.8],}
    param_settings['Cond4'] = { 'groupA_prop':0.50, 'params': [logit(0.10), 0.8, -0.5, -0.3, 0.8],}
    param_settings['Cond5'] = { 'groupA_prop':0.50, 'params': [logit(0.10), 1.0, -0.5, -0.3, 0.8],}
    param_settings['Cond6'] = { 'groupA_prop':0.50, 'params': [logit(0.10), 1.2, -0.5, -0.3, 0.8],}

     
    #local
    experimental_setup.experimental_loop(conditions=experimental_conditions,
                                        param_settings=param_settings, 
                                        base_save_folder='results', 
                                        num_runs=5, 
                                        total_sample_size=10, 
                                        verbose=2
                                )
    
    #hpc
    #experimental_setup.experimental_loop(conditions=experimental_conditions,
    #                                    param_settings=param_settings, 
    #                                    base_save_folder='results', 
    #                                    num_runs=10, 
    #                                    total_sample_size=100, 
    #                                    verbose=2
    #                            )
    

if __name__ == '__main__':
    main() 
