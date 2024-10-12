import pandas as pd
def preprocessing_df(df):
  df[df.columns[-1]] = pd.Categorical(df[df.columns[-1]]).codes
  df = df.drop(columns=df.columns[0])
  df = df.dropna()
  percentiles_99 = df.quantile(0.95)
  # Filter out records where values exceed the 99th percentile in any column
  df_filtered = df[(df <= percentiles_99).all(axis=1)]
  df = df_filtered
  df = df.drop(columns=df.columns[-1])
  df = df.dropna()
  return df
