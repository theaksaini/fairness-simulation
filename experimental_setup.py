import time
import random
import os
import pandas as pd
import pickle
from functools import partial
import traceback
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score,
    roc_auc_score)

import utils

def experimental_loop(conditions, param_settings, base_save_folder, num_runs, total_sample_size, verbose=0):
    
    for e, exp in enumerate(conditions):
        for r, reweight in enumerate([False, True]):

            save_folder = f"{base_save_folder}/{exp}_{reweight}"
            print("working on ")
            print(save_folder)
            time.sleep(random.random()*5)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            else:
                continue

            start_time = time.time()

            params = param_settings[exp]
            if verbose > 0:
                print("Beginning simulation.")

            results = {}

            # random seeds for each run
            seed = 100*e + r
            random.seed(seed)
            run_seeds = random.sample(range(num_runs * 3), num_runs)

            for (idx, r) in enumerate(run_seeds):
                if verbose == 2:
                    print('+----- Beginning run', idx + 1, '-----+')

                random.seed(r)
                x0 = np.ones(total_sample_size)                                # column of 1's for intercept term
                nA = int(np.round(total_sample_size*params['groupA_prop'], 0))           # size of group A (xA = 1)
                xA = list(np.ones(nA)) + list(np.zeros(total_sample_size-nA))  # binary sensitive feature (e.g., race)
                xC1 = np.random.normal(loc=0, scale=1, size=total_sample_size)      # continuous covariate from normal(0,1)
                xC2 = np.random.normal(loc=5, scale=2, size=total_sample_size)      # continuous covariate from normal(5,2)
                xC3 = np.random.randint(low=0, high=20, size=total_sample_size)     # integer covariate between 0 and 20
                XC3 = np.random.laplace(loc=0.0, scale=1.0, size=total_sample_size) # continuous covariate from laplace(0,1)
                X = np.column_stack((list(x0), list(xA), list(xC1), list(xC2), list(xC3)))

                # matrix multiplication to get linear combination, pass through an inv-logit function
                y = np.random.binomial(n=1, p =utils.inverse_logit(X @ params['params']), size=total_sample_size)

                df = pd.DataFrame({'y':y, 'xA':xA, 'xC1':xC1,' xC1':xC2, 'xC3':xC3})
                
                # @Amalia and Anil, feel free to add train-test split if you'd like
                model = LogisticRegression()
                x = df.drop(columns=['y'])

                # You must define calc_weights or replace with dummy weights (e.g., np.ones)
                w = utils.calc_weights(df, 'xA', 'y') if reweight else None

                if reweight:
                    model.fit(x, y, sample_weight=w)
                else:
                    model.fit(x, y)

                y_pred = model.predict(x)
                y_prob = model.predict_proba(x)[:, 1]

                # Performance metrics
                accuracy = accuracy_score(y, y_pred)
                precision = precision_score(y, y_pred, zero_division=0)
                recall = recall_score(y, y_pred)
                f1 = f1_score(y, y_pred)
                balanced_acc = balanced_accuracy_score(y, y_pred)
                auroc = roc_auc_score(y, y_prob)

                # Fairness metrics
                df['y_pred'] = y_pred
                df['y_true'] = y

                group_0 = df[df['xA'] == 0]
                group_1 = df[df['xA'] == 1]

                def tpr(group):
                    tp = ((group['y_true'] == 1) & (group['y_pred'] == 1)).sum()
                    fn = ((group['y_true'] == 1) & (group['y_pred'] == 0)).sum()
                    return tp / (tp + fn) if (tp + fn) > 0 else 0

                tpr_0 = tpr(group_0)
                tpr_1 = tpr(group_1)
                equal_opp_diff = abs(tpr_0 - tpr_1)

                ppr_0 = (group_0['y_pred'] == 1).mean()
                ppr_1 = (group_1['y_pred'] == 1).mean()
                dp_diff = abs(ppr_0 - ppr_1)
                fnr = ((y == 1) & (y_pred == 0)).sum() / (y == 1).sum() if (y == 1).sum() > 0 else 0
                fnr_0 = ((group_0['y_true'] == 1) & (group_0['y_pred'] == 0)).sum() / (group_0['y_true'] == 1).sum() if (group_0['y_true'] == 1).sum() > 0 else 0
                fnr_1 = ((group_1['y_true'] == 1) & (group_1['y_pred'] == 0)).sum() / (group_1['y_true'] == 1).sum() if (group_1['y_true'] == 1).sum() > 0 else 0
                dp_0 = ppr_0
                dp_1 = ppr_1


                results[idx + 1] = {
                    'seed': r,
                    'metrics': { 'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'balanced_accuracy': balanced_acc,
                        'auroc': auroc,
                        'equal_opportunity_diff': equal_opp_diff,
                        'demographic_parity_diff': dp_diff,
                        'fnr': fnr,
                        'fnr_group0': fnr_0,
                        'fnr_group1': fnr_1,
                        'dp_group0': dp_0,
                        'dp_group1': dp_1
                        },
                    'meta': {
                        'reweighted': reweight,
                        'groupA_prop': params['groupA_prop'],
                        'sample_size': total_sample_size
                    }
                }
            
            if verbose == 2:
                print('+----- Completed run',idx+1,'with random seed =', r,'-----+')
        
            flat_results = []
            for run_idx, data in results.items():
                row = {
                    'run_idx': run_idx,
                    'seed': data['seed'],
                    **data['metrics'],
                    **data['meta']
                }
                flat_results.append(row)

            results_df = pd.DataFrame(flat_results)

            if verbose > 0:
                print('Simulation complete')
                print('Total execution time:', time.time() - start_time)

            # Save the results
            with open(f"{save_folder}/scores.pkl", "wb") as f:
                pickle.dump(results_df, f)

            return 
                
    print("all finished")
