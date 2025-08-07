import pandas as pd

df = pd.read_parquet('data/FX.parquet')
print(df.head(5).to_string())
print(df.tail(5).to_string())