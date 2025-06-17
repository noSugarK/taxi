import pandas as pd
from datetime import datetime

class DataCleaner:
    def clean_data(self, df, min_long=113.8, max_long=114.5, min_lati=22.4, max_lati=22.8, max_speed=120):
        df_cleaned = df.copy()
        df_cleaned.drop_duplicates(inplace=True)
        try:
            df_cleaned['time'] = pd.to_datetime(df_cleaned['time'], format='%H:%M:%S')
        except Exception as e:
            print(f"时间格式转换错误: {e}")
        
        # 经纬度范围过滤
        df_cleaned = df_cleaned[(df_cleaned['long'] > min_long) & (df_cleaned['long'] < max_long)]
        df_cleaned = df_cleaned[(df_cleaned['lati'] > min_lati) & (df_cleaned['lati'] < max_lati)]
        
        # 速度异常过滤
        df_cleaned = df_cleaned[df_cleaned['speed'] < max_speed]
        
        df_cleaned = df_cleaned.dropna(subset=['time', 'long', 'lati', 'status'])
        return df_cleaned