import pandas as pd

DATA_PATH = '/Users/gurkeinan/semester6/Causal-Inference/Project/code/data/processed_data.csv'
df = pd.read_csv(DATA_PATH)
print(df.head())
