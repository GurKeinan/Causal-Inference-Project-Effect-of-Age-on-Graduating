from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

from utils import read_and_transform_data, calculate_propensity_scores, bootstrap_confidence_intervals

def fit_outcome_models(X, t, y):
    X_1 = X[t == 1]
    X_0 = X[t == 0]
    y_1 = y[t == 1]
    y_0 = y[t == 0]

    model_1 = RandomForestClassifier(random_state=42)
    model_1.fit(X_1, y_1)

    model_0 = RandomForestClassifier(random_state=42)
    model_0.fit(X_0, y_0)

    return model_1, model_0


def calculate_measures_doubly_robust(X, t, y):
    n = len(y)

    e = calculate_propensity_scores(X, t)
    model_1, model_0 = fit_outcome_models(X, t, y)

    y_pred_all_1 = model_1.predict(X)
    y_pred_all_0 = model_0.predict(X)

    g_1_score = y_pred_all_1 + (t / e) * (y - y_pred_all_1)
    g_0_score = y_pred_all_0 + ((1 - t) / (1 - e)) * (y - y_pred_all_0)

    ATE = np.sum(g_1_score) / n - np.sum(g_0_score) / n
    ATT = np.sum(t * y - ((t - e) * y_pred_all_0 / (1 - e))) / np.sum(t)
    ATC = np.sum((1 - e) * t * y / e - ((t - e) * y_pred_all_1 / e) - ((1 - t) * y)) / np.sum(1 - t)

    return ATE, ATT, ATC

DATA_PATH = '/Users/gurkeinan/semester6/Causal-Inference/Project/code/data/processed_data.csv'

if __name__ == '__main__':
    X, t, y = read_and_transform_data(DATA_PATH)

    # Calculate point estimates
    ATE, ATT, ATC = calculate_measures_doubly_robust(X, t, y)
    print(f'ATE: {ATE:.4f}, ATT: {ATT:.4f}, ATC: {ATC:.4f}')

    # Calculate confidence intervals
    ATE_CI, ATT_CI, ATC_CI, bootstrap_ATE, bootstrap_ATT, bootstrap_ATC = bootstrap_confidence_intervals(X, t, y, calculate_measures_doubly_robust)
    print(f'ATE 95% CI: {ATE_CI}, ATT 95% CI: {ATT_CI}, ATC 95% CI: {ATC_CI}')