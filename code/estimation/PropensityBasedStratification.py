from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

from utils import read_and_transform_data, calculate_propensity_scores, bootstrap_confidence_intervals


def calculate_measures_propensity_score_stratification(X, t, y, num_strata=5):
    # Calculate propensity scores
    e = calculate_propensity_scores(X, t)

    # Create strata based on the propensity scores
    strata = pd.qcut(e, num_strata, labels=False)

    # Calculate the ATE, ATT, and ATC for each stratum
    results = []
    for i in range(num_strata):
        mask = (strata == i)
        treated_mask = mask & (t == 1)
        control_mask = mask & (t == 0)

        # Calculate mean outcomes in each stratum for treated and control groups
        treated_mean = y[treated_mask].mean()
        control_mean = y[control_mask].mean()

        # Calculate treatment effect within each stratum
        strata_effect = treated_mean - control_mean

        # Collect results
        results.append({
            'strata': i,
            'n_treated': treated_mask.sum(),
            'n_control': control_mask.sum(),
            'n_total': mask.sum(),
            'effect': strata_effect
        })

    # Convert to DataFrame for easier processing
    strata_df = pd.DataFrame(results)

    # Calculate overall ATE, ATT, and ATC

    # ATE: Weighted by total individuals in each stratum
    ATE = np.sum(strata_df['effect'] * strata_df['n_total'] / strata_df['n_total'].sum())

    # ATT: Weighted by number of treated individuals in each stratum
    ATT = np.sum(strata_df['effect'] * strata_df['n_treated'] / strata_df['n_treated'].sum())

    # ATC: Weighted by number of control individuals in each stratum
    ATC = np.sum(strata_df['effect'] * strata_df['n_control'] / strata_df['n_control'].sum())

    return ATE, ATT, ATC


# def bootstrap_confidence_intervals_propensity_score_stratification(X, t, y, num_bootstrap=1000, ci_level=95):
#     bootstrap_ATE = []
#     bootstrap_ATT = []
#     bootstrap_ATC = []
#
#     # Perform bootstrap sampling
#     for _ in range(num_bootstrap):
#         # Create a bootstrap sample (resample with replacement)
#         indices = np.random.choice(len(X), size=len(X), replace=True)
#         X_bootstrap = X.iloc[indices]
#         t_bootstrap = t.iloc[indices]
#         y_bootstrap = y.iloc[indices]
#
#         # Compute the ATE, ATT, and ATC for this bootstrap sample
#         ATE, ATT, ATC = calculate_measures_propensity_score_stratification(X_bootstrap, t_bootstrap, y_bootstrap)
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
    ATE, ATT, ATC = calculate_measures_propensity_score_stratification(X, t, y)
    print(f'ATE: {ATE:.4f}, ATT: {ATT:.4f}, ATC: {ATC:.4f}')

    # Calculate confidence intervals
    ATE_CI, ATT_CI, ATC_CI, bootstrap_ATE, bootstrap_ATT, bootstrap_ATC = bootstrap_confidence_intervals(X, t, y, calculate_measures_propensity_score_stratification)
    print(f'ATE 95% CI: {ATE_CI}, ATT 95% CI: {ATT_CI}, ATC 95% CI: {ATC_CI}')