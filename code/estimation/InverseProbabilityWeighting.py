from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

from utils import read_and_transform_data, calculate_propensity_scores, bootstrap_confidence_intervals


def calculate_measures_IPW(X, t, y):
    n = len(y)
    e = calculate_propensity_scores(X, t)
    ATE = sum(y * t / e) / n - sum(y * (1 - t) / (1 - e)) / n
    ATT = sum(y * t) / sum(t) - sum(y * (1 - t) * e / (1 - e)) / sum((1 - t) * e / (1 - e))
    ATC = sum(y * t * (1 - e) / e) / sum(t * (1 - e) / e) - sum(y * (1 - t)) / sum(1 - t)

    return ATE, ATT, ATC


# def bootstrap_confidence_intervals_IPW(X, t, y, num_bootstrap=1000, ci_level=95):
#     bootstrap_ATE = []
#     bootstrap_ATT = []
#     bootstrap_ATC = []
#
#     # Combine X, t, and y into a single dataframe for easier resampling
#     data = pd.concat([X, pd.Series(t, name='Adult'), pd.Series(y, name='Target')], axis=1)
#
#     # Perform bootstrap sampling
#     for _ in range(num_bootstrap):
#         # Create a bootstrap sample (resample with replacement)
#         bootstrap_sample = data.sample(n=len(data), replace=True)
#
#         # Split the bootstrap sample back into X, t, and y
#         X_bootstrap = bootstrap_sample.drop(['Adult', 'Target'], axis=1)
#         t_bootstrap = bootstrap_sample['Adult']
#         y_bootstrap = bootstrap_sample['Target']
#
#         # Calculate propensity scores for this bootstrap sample
#         e_bootstrap = calculate_propensity_scores(X_bootstrap, t_bootstrap)
#
#         # Compute the ATE, ATT, and ATC for this bootstrap sample
#         ATE, ATT, ATC = calculate_measures_IPW(y_bootstrap, t_bootstrap, e_bootstrap)
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
    X, t, y = read_and_transform_data(DATA_PATH)

    # Calculate point estimates
    ATE, ATT, ATC = calculate_measures_IPW(X, t, y)
    print(f'ATE: {ATE:.4f}, ATT: {ATT:.4f}, ATC: {ATC:.4f}')

    # Calculate confidence intervals
    ATE_CI, ATT_CI, ATC_CI, bootstrap_ATE, bootstrap_ATT, bootstrap_ATC = bootstrap_confidence_intervals(X, t, y, calculate_measures_IPW)
    print(f'ATE 95% CI: {ATE_CI}, ATT 95% CI: {ATT_CI}, ATC 95% CI: {ATC_CI}')
