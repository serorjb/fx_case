import pandas as pd
import numpy as np
from pathlib import Path
from scipy.interpolate import PchipInterpolator
import plotly.graph_objects as go
import matplotlib.dates as mdates
import os

# Load FRED files
def load_fred(filename, label):
    df = pd.read_csv(filename)
    df.columns = ['date', label]
    df['date'] = pd.to_datetime(df['date'])
    df[label] = pd.to_numeric(df[label], errors='coerce') / 100
    return df.set_index('date')

# Load all series
data_dir = Path("data/FRED")
df_1m = load_fred(data_dir / "DTB4WK.csv", "1M")
df_3m = load_fred(data_dir / "DTB3.csv", "3M")
df_6m = load_fred(data_dir / "DTB6.csv", "6M")
df_1y = load_fred(data_dir / "DTB1YR.csv", "1Y")

# Merge and forward fill
df_rates = df_1m.join([df_3m, df_6m, df_1y], how='outer').sort_index().ffill()

# Define FX option target tenors (in years)
target_tenors = {
    '1W': 1/52,
    '2W': 2/52,
    '3W': 3/52,
    '1M': 1/12,
    '2M': 2/12,
    '3M': 3/12,
    '4M': 4/12,
    '6M': 6/12,
    '9M': 9/12,
    '1Y': 1.0
}

records = []

# Iterate over daily yield curves
for date, row in df_rates.iterrows():
    available = row.dropna()
    if len(available) < 2:
        continue

    tenors = []
    dfs = []
    for label, r in available.items():
        T = int(label.strip("MY")) / 12 if 'M' in label else float(label.strip("Y"))
        tenors.append(T)
        dfs.append(np.exp(-r * T))

    # Interpolate in discount factor space
    try:
        interpolator = PchipInterpolator(tenors, dfs, extrapolate=True)
    except Exception:
        continue

    for label, T in target_tenors.items():
        df_val = float(interpolator(T))

        # Clip DF to valid arbitrage-free range
        df_val = np.clip(df_val, 1e-6, 1.0)

        # Hull-style rate extraction with Taylor for very short T
        if T < 0.025:
            r = 1.0 - df_val  # Taylor approx
        else:
            r = -np.log(df_val) / T

        # Cap very high/low short-end rates
        if T <= 0.1:  # ~1M or shorter
            r = np.clip(r, -0.05, 0.25)

        records.append({
            'date': date,
            'tenor': label,
            'tenor_years': T,
            'discount_factor': df_val,
            'interpolated_rate': r
        })

# Final DataFrame
df_discount = pd.DataFrame(records)
df_discount = df_discount.sort_values(['date', 'tenor'])

# Save to Parquet
df_discount.to_parquet("data/discount_curves.parquet", index=False)
print("✅ Discount curve saved to data/discount_curves.parquet")

# -----------------------------
# 3D Plot: Yield Curves Over Time
# -----------------------------
os.makedirs("plots", exist_ok=True)
pivot = df_discount.pivot_table(index="date", columns="tenor_years", values="interpolated_rate")
dates = mdates.date2num(pivot.index)
tenors = pivot.columns.values
rates = pivot.values

fig = go.Figure(data=[go.Surface(
    x=tenors,
    y=pivot.index.strftime('%Y-%m-%d'),
    z=rates,
    colorscale='Viridis'
)])

fig.update_layout(
    title="Yield Curves Over Time",
    scene=dict(
        xaxis_title="Tenor (years)",
        yaxis_title="Date",
        zaxis_title="Interpolated Rate"
    ),
    autosize=True
)

fig.write_html("plots/discount_curves_3d.html")
print("✅ 3D plot saved to plots/discount_curves_3d.html")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.dates as mdates
import numpy as np

# Prepare data for 3D plot
pivot = df_discount.pivot_table(index="date", columns="tenor_years", values="interpolated_rate")
dates = mdates.date2num(pivot.index)
tenors = pivot.columns.values
rates = pivot.values

X, Y = np.meshgrid(tenors, dates)
Z = rates

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

ax.set_xlabel("Tenor (years)")
ax.set_ylabel("Date")
ax.set_zlabel("Interpolated Rate")
ax.set_title("Yield Curves Over Time")

ax.yaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
fig.autofmt_xdate()

plt.colorbar(surf, shrink=0.5, aspect=10, label="Interpolated Rate")
plt.tight_layout()
plt.savefig("plots/discount_curves_3d.png")
plt.close()
print("✅ PNG plot saved to plots/discount_curves_3d.png")
