import pandas as pd
def preprocessing_df(df):
  df['device'] = pd.Categorical(df['device']).codes
  df = df.drop(columns=[['ID', 'device']])
  df = df.dropna()
  df1 = df.drop(columns=['label'])
  df2 = df['label']
  percentiles_99 = df1.quantile(0.95)
  # Filter out records where values exceed the 99th percentile in any column
  df_filtered = df1[(df1 <= percentiles_99).all(axis=1)]
  df = df_filtered
  df['label'] = df2
  #df = df.drop(columns=['device'])
  df = df.dropna()
  df['label'] = pd.Categorical(df['label']).codes
  return df
