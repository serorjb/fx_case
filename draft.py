import pandas as pd

# Load risk-free rate data
df_rf = pd.read_csv("data/FRED_DGS10.csv")
df_rf.columns = ['date', 'rf_rate']
df_rf['date'] = pd.to_datetime(df_rf['date'])
df_rf.set_index('date', inplace=True)

# Convert percent to decimal
df_rf['rf_rate'] = df_rf['rf_rate'].replace('.', float('nan'))  # Some entries may be '.'
df_rf['rf_rate'] = pd.to_numeric(df_rf['rf_rate'], errors='coerce') / 100

# Forward-fill missing dates (weekends/holidays)
df_rf = df_rf.resample('D').ffill()

# Load FX data
df_fx = pd.read_parquet("data/fx.parquet")
df_fx.index = pd.to_datetime(df_fx.index)

# Merge risk-free rate
df_fx = df_fx.merge(df_rf, how='left', left_index=True, right_index=True)
df_fx['rf_rate'] = df_fx['rf_rate'].fillna(method='ffill')

# Save merged version if needed
df_fx.to_parquet("data/fx_merged.parquet")
print(df_fx)