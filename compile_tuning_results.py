# Combining tuning results

import pandas as pd
import numpy as np
import os

data_path = 'tuning_results/'
case_suffix = '_soil.xlsx'
compiled_outpath = os.path.join(data_path, 'NN_20240801_tuning_soil.xlsx')

files = [f for f in os.listdir(data_path) if case_suffix in f]
dfs = [pd.read_excel(f) for f in files]
#dfs = [df.rename(columns={'tmu': 'lr'}) if 'tmu' in df.columns else df for df in dfs]
#to_drop = np.arange(0,92,1)
#dfs[0] = dfs[0].drop(to_drop)
df = pd.concat(dfs, ignore_index=True)
df = df.drop_duplicates()
df.to_excel(compiled_outpath)

grouped_df = df.groupby(['lr', 'hidden_size', 'hidden_layer', 'batch_size', 'optim', 'criterion', 'Dataset']).mean()

test_df = grouped_df.loc[grouped_df.index.get_level_values('Dataset') == 'Test']
min_rmse_idx = test_df['RMSE'].idxmin()
min_rmse_row = test_df.loc[min_rmse_idx]
print(min_rmse_row)

max_rmse_idx = test_df['RMSE'].idxmax()
max_rmse_row = test_df.loc[max_rmse_idx ]
print(max_rmse_row)

group_values = min_rmse_idx[:-1]

val_df = grouped_df.loc[(grouped_df.index.get_level_values('Dataset') == 'Val')]

# Further filter val_df to include only the rows where the index matches the group values
val_df_filtered = val_df.loc[val_df.index.get_level_values('lr') == group_values[0]]
val_df_filtered = val_df_filtered.loc[val_df_filtered.index.get_level_values('hidden_size') == group_values[1]]
val_df_filtered = val_df_filtered.loc[val_df_filtered.index.get_level_values('hidden_layer') == group_values[2]]
val_df_filtered = val_df_filtered.loc[val_df_filtered.index.get_level_values('batch_size') == group_values[3]]
val_df_filtered = val_df_filtered.loc[val_df_filtered.index.get_level_values('optim') == group_values[4]]
val_df_filtered = val_df_filtered.loc[val_df_filtered.index.get_level_values('criterion') == group_values[5]]