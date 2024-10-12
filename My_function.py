import pandas as pd
def preprocessing_df(df):
  df[11] = pd.Categorical(df[11]).codes
  df = df.drop(columns=[0])
  df = df.dropna()
  percentiles_99 = df.quantile(0.95)
  # Filter out records where values exceed the 99th percentile in any column
  df_filtered = df[(df <= percentiles_99).all(axis=1)]
  df = df_filtered
  df = df.drop(columns=[11])
  df = df.dropna()
  return df
