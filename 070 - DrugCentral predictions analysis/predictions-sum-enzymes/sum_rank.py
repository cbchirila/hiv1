import pandas as pd

csv_files = [
    'abc-pred-sum-PR.csv',
    'cnn-pred-sum-PR.csv',
    'dtc-pred-sum-PR.csv',
    'gpc-pred-sum-PR.csv',
    'knn-pred-sum-PR.csv',
    'mlp-pred-sum-PR.csv',
    'rfc-pred-sum-PR.csv',
    'svc-pred-sum-PR.csv',
    ]

for file in csv_files:

    df = pd.read_csv(file)
    
    df_list = [pd.read_csv(file) for file in csv_files]

    df = pd.concat(df_list, ignore_index=True)

    df_sum = df.groupby('title', as_index=False)['rank'].sum()

    df_sorted = df_sum.sort_values(by='rank', ascending=True)
    
    df_sorted.to_csv('sum_rank_cnn_knn_mlp_rfc_PR.csv', index=False)

print(df_sorted)
