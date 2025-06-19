import pandas as pd
from datetime import datetime

class DataCleaner:
    def load_data(self, file_path):
        if 'TaxiData1e6.csv' in file_path:
            df = pd.read_csv(file_path, header=1)
            df = df.iloc[:, 1:]
            df.columns = ['id', 'time', 'long', 'lati', 'status', 'speed']
        else:
            df = pd.read_csv(file_path, header=None, names=['id', 'time', 'long', 'lati', 'status', 'speed'])
        return df

    def clean_data(self, df, min_long=113.85, max_long=114.6, min_lati=22.4, max_lati=22.8, max_speed=120):
        df_cleaned = df.copy()
        
        # 确保按id和time排序，这是后续处理的基础
        df_cleaned['time'] = pd.to_datetime(df_cleaned['time'], format='%H:%M:%S')
        df_cleaned = df_cleaned.sort_values(by=['id', 'time']).reset_index(drop=True)

        # 找到id和time都相同的重复记录
        duplicates_mask = df_cleaned.duplicated(subset=['id', 'time'], keep=False)

        # 经纬度范围过滤
        df_cleaned = df_cleaned[(df_cleaned['long'] > min_long) & (df_cleaned['long'] < max_long)]
        df_cleaned = df_cleaned[(df_cleaned['lati'] > min_lati) & (df_cleaned['lati'] < max_lati)]
        # 速度异常过滤
        df_cleaned = df_cleaned[df_cleaned['speed'] < max_speed]
        df_cleaned = df_cleaned.dropna(subset=['time', 'long', 'lati', 'status'])
        # 处理重复记录
        processed_df_list = []
        for (id_val, time_val), group in df_cleaned[duplicates_mask].groupby(['id', 'time']):
            unique_statuses = group['status'].nunique()
            if unique_statuses == 1:
                # status相同，保留第一条
                processed_df_list.append(group.iloc[[0]])
            else:
                # status不同，需要根据状态变化决定保留哪条
                if 1 in group['status'].values:
                    processed_df_list.append(group[group['status'] == 1].iloc[[0]])
                else:
                    processed_df_list.append(group.iloc[[0]])
        
        # 将处理过的重复记录和非重复记录合并
        df_non_duplicates = df_cleaned[~duplicates_mask]
        if processed_df_list:
            df_processed_duplicates = pd.concat(processed_df_list)
            df_cleaned = pd.concat([df_non_duplicates, df_processed_duplicates]).sort_values(by=['id', 'time']).reset_index(drop=True)
        else:
            df_cleaned = df_non_duplicates.sort_values(by=['id', 'time']).reset_index(drop=True)

        # 为每个id创建一个组，并在组内进行状态比较
        df_cleaned['prev_status'] = df_cleaned.groupby('id')['status'].shift(1)
        df_cleaned['next_status'] = df_cleaned.groupby('id')['status'].shift(-1)
        # 排除NaN值（即序列的开头和结尾）
        abnormal_status_mask = (
            (df_cleaned['status'] != df_cleaned['prev_status']) & 
            (df_cleaned['status'] != df_cleaned['next_status']) & 
            (df_cleaned['prev_status'].notna()) & # 确保不是序列的第一个
            (df_cleaned['next_status'].notna())    # 确保不是序列的最后一个
        )
        # 移除异常状态的记录
        df_cleaned = df_cleaned[~abnormal_status_mask].reset_index(drop=True)
        # 最终确保按id和time排序
        df_cleaned = df_cleaned.sort_values(by=['id', 'time']).reset_index(drop=True)
        # 移除辅助列
        df_cleaned = df_cleaned.drop(columns=['prev_status', 'next_status'])
        return df_cleaned