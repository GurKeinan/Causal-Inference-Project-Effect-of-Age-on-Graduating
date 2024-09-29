import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_and_preprocess_data(file_path):
    # Load the data
    df = pd.read_csv(file_path)

    # Define the treatment: 1 if age at enrollment > 21, else 0
    df['treatment'] = (df['Age at enrollment'] > 21).astype(int)

    # Define the outcome: 1 if graduated, 0 otherwise
    df['outcome'] = (df['Target'] == 'Graduate').astype(int)

    # Select features for propensity score model
    features = ['Previous qualification (grade)', 'Admission grade', 'Mother\'s qualification','Father\'s qualification', 'Gender', 'Nacionality', 'Displaced', 'Scholarship holder', 'Tuition fees up to date', 'Unemployment rate', 'Inflation rate', 'GDP']

    return df, features

def estimate_propensity_scores(df, features):
    # Create preprocessing pipeline
    numeric_features = ['Previous qualification (grade)', 'Admission grade', 'Unemployment rate', 'Inflation rate', 'GDP']
    categorical_features = [f for f in features if f not in numeric_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ])

    # Create pipeline with preprocessor and logistic regression
    propensity_model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression())
    ])

    # Fit the propensity score model
    propensity_model.fit(df[features], df['treatment'])

    # Calculate propensity scores
    propensity_scores = propensity_model.predict_proba(df[features])[:, 1]

    return propensity_scores

def match_samples(df, propensity_scores, caliper=0.2):
    treated = df[df['treatment'] == 1]
    control = df[df['treatment'] == 0]

    matches = []
    for i, treated_unit in treated.iterrows():
        treated_ps = propensity_scores[i]

        # Find control units within caliper
        potential_matches = control[(propensity_scores[control.index] >= treated_ps - caliper) &
                                    (propensity_scores[control.index] <= treated_ps + caliper)]
        if not potential_matches.empty:
            # Find the closest match
            closest_match = potential_matches.iloc[np.abs(propensity_scores[potential_matches.index] - treated_ps).argsort()[0]]
            matches.append((treated_unit, closest_match))

    return matches

def calculate_ate(matches):
    treatment_effects = [treated['outcome'] - control['outcome'] for treated, control in matches]
    ate = np.mean(treatment_effects)
    se = np.std(treatment_effects) / np.sqrt(len(treatment_effects))
    ci_lower, ci_upper = ate - 1.96 * se, ate + 1.96 * se

    return ate, ci_lower, ci_upper

def main():
    file_path = '/Users/gurkeinan/semester6/Causal-Inference/Project/code/data.csv'  # Replace with your actual file path
    df, features = load_and_preprocess_data(file_path)

    propensity_scores = estimate_propensity_scores(df, features)
    matches = match_samples(df, propensity_scores)

    ate, ci_lower, ci_upper = calculate_ate(matches)

    print(f"Estimated ATE: {ate:.4f}")
    print(f"95% Confidence Interval: ({ci_lower:.4f}, {ci_upper:.4f})")

if __name__ == "__main__":
    main()