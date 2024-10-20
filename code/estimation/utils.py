import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


def read_and_transform_data(data_path):
    data = pd.read_csv(data_path)
    n = len(data)

    t = data['Adult']
    y = data['Target']
    X = data.drop(columns=['Adult', 'Target'])

    numerical_columns = ['Previous qualification (grade)', 'Admission grade', 'Unemployment rate', 'Inflation rate', 'GDP']
    categorical_columns = list(set(X.columns) - set(numerical_columns))

    # Scaling numerical columns
    scaler = StandardScaler()
    X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

    # One-hot encoding for categorical columns
    X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

    # remove from data the column Application mode_39
    X = X.drop(columns=['Application mode_39'])

    return X, t, y

def calculate_propensity_scores(X, t):
    propensity_model = RandomForestClassifier(random_state=42)
    propensity_model.fit(X, t)
    e = propensity_model.predict_proba(X)[:, 1]

    # Clip propensity scores to avoid extreme values
    epsilon = 1e-5
    e = np.clip(e, epsilon, 1 - epsilon)

    return e

def bootstrap_confidence_intervals(X, t, y, calculate_measures, num_bootstrap=1000, ci_level=95, **kwargs):
    """
    Perform bootstrap resampling to calculate confidence intervals for ATE, ATT, and ATC.

    Parameters:
    X (pd.DataFrame): Feature matrix
    t (pd.Series): Treatment assignments
    y (pd.Series): Outcome variable
    calculate_measures (function): Function to calculate ATE, ATT, and ATC for a given sample
    num_bootstrap (int): Number of bootstrap iterations
    ci_level (float): Confidence interval level (e.g., 95 for 95% CI)
    **kwargs: Additional keyword arguments to pass to calculate_measures function

    Returns:
    tuple: ATE_CI, ATT_CI, ATC_CI, bootstrap_ATE, bootstrap_ATT, bootstrap_ATC
    """
    bootstrap_ATE = []
    bootstrap_ATT = []
    bootstrap_ATC = []

    # Combine X, t, and y into a single dataframe for easier resampling
    data = pd.concat([X, t, y], axis=1)

    # Perform bootstrap sampling
    for _ in range(num_bootstrap):
        # Create a bootstrap sample (resample with replacement)
        bootstrap_sample = data.sample(n=len(data), replace=True)

        # Split the bootstrap sample back into X, t, and y
        X_bootstrap = bootstrap_sample.iloc[:, :-2]
        t_bootstrap = bootstrap_sample.iloc[:, -2]
        y_bootstrap = bootstrap_sample.iloc[:, -1]

        # Calculate measures for this bootstrap sample
        ATE, ATT, ATC = calculate_measures(X_bootstrap, t_bootstrap, y_bootstrap, **kwargs)

        # Store the results
        bootstrap_ATE.append(ATE)
        bootstrap_ATT.append(ATT)
        bootstrap_ATC.append(ATC)

    # Calculate percentiles for confidence intervals
    lower_percentile = (100 - ci_level) / 2
    upper_percentile = 100 - lower_percentile

    ATE_CI = np.percentile(bootstrap_ATE, [lower_percentile, upper_percentile])
    ATT_CI = np.percentile(bootstrap_ATT, [lower_percentile, upper_percentile])
    ATC_CI = np.percentile(bootstrap_ATC, [lower_percentile, upper_percentile])

    return ATE_CI, ATT_CI, ATC_CI, bootstrap_ATE, bootstrap_ATT, bootstrap_ATC