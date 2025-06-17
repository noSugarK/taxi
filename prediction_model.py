import pandas as pd
from datetime import datetime, timedelta

class PredictionModel:
    def __init__(self):
        pass

    def predict_demand(self, historical_data: pd.DataFrame, time_period: str):
        """
        预测特定时间和地点的乘客需求。
        historical_data: 包含历史订单数据，至少包括时间、上车地点等信息。
        time_period: 预测的时间段，例如 'hourly', 'daily'。
        """
        print(f"正在预测 {time_period} 的乘客需求...")
        
        if historical_data.empty or 'O_time' not in historical_data.columns:
            print("历史数据为空或缺少'O_time'列，无法进行需求预测。")
            return pd.DataFrame(columns=['time_unit', 'demand'])

        # 确保 'O_time' 是 datetime 类型
        historical_data['O_time'] = pd.to_datetime(historical_data['O_time'])

        if time_period == 'hourly':
            historical_data['time_unit'] = historical_data['O_time'].dt.hour
            demand_by_unit = historical_data.groupby('time_unit').size().reset_index(name='demand')
            # 填充所有小时，确保0-23小时都有数据，没有数据的填充0
            all_hours = pd.DataFrame({'time_unit': range(24)})
            demand_by_unit = pd.merge(all_hours, demand_by_unit, on='time_unit', how='left').fillna(0)
            demand_by_unit['demand'] = demand_by_unit['demand'].astype(int)
            print("按小时预测需求完成。")
            return demand_by_unit
        elif time_period == 'daily':
            historical_data['time_unit'] = historical_data['O_time'].dt.date
            demand_by_unit = historical_data.groupby('time_unit').size().reset_index(name='demand')
            print("按天预测需求完成。")
            return demand_by_unit
        else:
            print(f"不支持的时间粒度: {time_period}。目前只支持 'hourly' 和 'daily'。")
            return pd.DataFrame(columns=['time_unit', 'demand'])

    def predict_eta(self, start_location: tuple, end_location: tuple, current_time: datetime):
        """
        预测出租车从起点到终点的预计到达时间（ETA）。
        start_location: 起点坐标 (经度, 纬度)。
        end_location: 终点坐标 (经度, 纬度)。
        current_time: 当前时间。
        """
        print(f"正在预测从 {start_location} 到 {end_location} 的ETA...")
        
        # 引入DataAnalyzer来计算距离
        from data_analyzer import DataAnalyzer
        analyzer = DataAnalyzer()
        distance_km = analyzer.haversine(start_location[0], start_location[1], end_location[0], end_location[1])
        
        if distance_km < 0.01: # 如果距离非常近，ETA接近0
            return "不足 1 分钟"

        # 根据时间调整平均速度 (km/h)
        # 这是一个简化的模型，实际应考虑实时交通、路况等
        hour = current_time.hour
        if 6 <= hour < 9: # 早高峰
            avg_speed_kmh = 20
        elif 17 <= hour < 20: # 晚高峰
            avg_speed_kmh = 18
        elif 22 <= hour or hour < 5: # 夜间
            avg_speed_kmh = 40
        else: # 平峰
            avg_speed_kmh = 30
            
        avg_speed_mps = avg_speed_kmh * 1000 / 3600 # 转换为米/秒

        time_seconds = (distance_km * 1000) / avg_speed_mps
        eta_minutes = time_seconds / 60
        
        return f"{eta_minutes:.2f} 分钟"