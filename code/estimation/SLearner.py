import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from utils import read_and_transform_data, bootstrap_confidence_intervals

def calculate_measures_s_learner(X, t, y):
    model = RandomForestClassifier(random_state=42)
    model.fit(pd.concat([X, t], axis=1), y)

    # Predict outcomes for all individuals under treatment condition
    Xt = pd.concat([X, t], axis=1)
    Xt_treated = Xt.copy()
    Xt_treated['Adult'] = 1
    y_pred_treated = model.predict_proba(Xt_treated)[:, 1]
    
    # Predict outcomes for all individuals under control condition
    Xt_control = Xt.copy()
    Xt_control['Adult'] = 0
    y_pred_control = model.predict_proba(Xt_control)[:, 1]
    
    # Calculate individual treatment effects
    individual_effects = y_pred_treated - y_pred_control
    
    # Calculate ATE
    ATE = np.mean(individual_effects)
    
    # Calculate ATT
    ATT = np.mean(individual_effects[Xt['Adult'] == 1])
    
    # Calculate ATC
    ATC = np.mean(individual_effects[Xt['Adult'] == 0])
    
    return ATE, ATT, ATC

# def bootstrap_confidence_intervals_s_learner(X, t, y, model, num_bootstrap=1000, ci_level=95):
#     bootstrap_ATE = []
#     bootstrap_ATT = []
#     bootstrap_ATC = []
#
#     # Combine X, t, and y into a single dataframe for easier resampling
#     data = pd.concat([X, pd.Series(t, name='Adult'), pd.Series(y, name='Target')], axis=1)
#
#     # Perform bootstrap sampling
#     for i in range(num_bootstrap):
#         # Create a bootstrap sample (resample with replacement)
#         bootstrap_sample = data.sample(n=len(data), replace=True)
#
#         # Split the bootstrap sample back into X, t, and y
#         X_bootstrap = bootstrap_sample.drop(['Adult', 'Target'], axis=1)
#         t_bootstrap = bootstrap_sample['Adult']
#         y_bootstrap = bootstrap_sample['Target']
#
#         # Refit the model on the bootstrap sample
#         bootstrap_model = clone(model)  # Create a fresh copy of the model
#         bootstrap_model.fit(pd.concat([X_bootstrap, t_bootstrap], axis=1), y_bootstrap)
#
#         # Compute the ATE, ATT, and ATC for this bootstrap sample
#         ATE, ATT, ATC = calculate_measures_s_learner(X_bootstrap, t_bootstrap, bootstrap_model)
#
#         # Store the results
#         bootstrap_ATE.append(ATE)
#         bootstrap_ATT.append(ATT)
#         bootstrap_ATC.append(ATC)
#
#     # Calculate percentiles for confidence intervals
#     lower_percentile = (100 - ci_level) / 2
#     upper_percentile = 100 - lower_percentile
#
#     ATE_CI = np.percentile(bootstrap_ATE, [lower_percentile, upper_percentile])
#     ATT_CI = np.percentile(bootstrap_ATT, [lower_percentile, upper_percentile])
#     ATC_CI = np.percentile(bootstrap_ATC, [lower_percentile, upper_percentile])
#
#     return ATE_CI, ATT_CI, ATC_CI, bootstrap_ATE, bootstrap_ATT, bootstrap_ATC

DATA_PATH = '/Users/gurkeinan/semester6/Causal-Inference/Project/code/data/processed_data.csv'

if __name__ == '__main__':
    data = pd.read_csv(DATA_PATH)
    X, t, y = read_and_transform_data(DATA_PATH)
    
    # calculate ATE, ATT, ATC
    ATE, ATT, ATC = calculate_measures_s_learner(X, t, y)
    print(f'ATE: {ATE}, ATT: {ATT}, ATC: {ATC}')
    
    # calculate confidence intervals
    ATE_CI, ATT_CI, ATC_CI, bootstrap_ATE, bootstrap_ATT, bootstrap_ATC = bootstrap_confidence_intervals(X, t, y, calculate_measures_s_learner)
    print(f'ATE 95% CI: {ATE_CI}, ATT 95% CI: {ATT_CI}, ATC 95% CI: {ATC_CI}')
