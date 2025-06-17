import matplotlib.pyplot as plt
from datetime import datetime

# 数据可视化模块
class DataVisualizer:
    def __init__(self):
        self.default_figsize = (8, 6) # 设置默认图片大小
    
    def plot_gps_points(self, df):
        """绘制GPS点分布图"""
        fig, ax = plt.subplots(figsize=self.default_figsize)
        scatter = ax.scatter(df['long'], df['lati'], c=df['status'], 
                           cmap='coolwarm', alpha=0.6, marker='o', s=10)
        
        plt.title('出租车GPS点分布')
        plt.xlabel('经度')
        plt.ylabel('纬度')
        cbar = plt.colorbar(scatter)
        cbar.set_label('状态 (0=空车, 1=载客)')
        
        return fig
    
    def plot_hotspots(self, hotspots_df):
        """绘制热门上客点热力图"""
        fig, ax = plt.subplots(figsize=self.default_figsize)
        
        # 使用散点图表示热点，点的大小表示热力值
        scatter = ax.scatter(hotspots_df['lng'], hotspots_df['lat'], 
                           s=hotspots_df['count']*2, # 点大小与热力值成正比
                           alpha=0.7, 
                           c=hotspots_df['count'], 
                           cmap='hot_r')
        
        plt.title('热门上客点分布')
        plt.xlabel('经度')
        plt.ylabel('纬度')
        cbar = plt.colorbar(scatter)
        cbar.set_label('上客次数')
        
        return fig
    
    def plot_time_distribution(self, time_dist_df):
        """绘制时间分布图"""
        fig, ax = plt.subplots(figsize=(10, 6)) # 调整为更适合时间分布的尺寸
        
        # 转换时间格式为字符串，便于显示
        time_labels = time_dist_df['O_time'].dt.strftime('%H:%M')
        
        ax.bar(range(len(time_dist_df)), time_dist_df['count'], color='skyblue')
        ax.set_xticks(range(0, len(time_dist_df), max(1, len(time_dist_df)//24)))
        ax.set_xticklabels(time_labels[::max(1, len(time_dist_df)//24)], rotation=45)
        
        plt.title('乘客打车时间分布')
        plt.xlabel('时间')
        plt.ylabel('打车次数')
        plt.tight_layout()
        
        return fig
    
    def plot_speed_by_hour(self, speed_df):
        """绘制每小时平均速度图"""
        fig, ax = plt.subplots(figsize=self.default_figsize)
        
        ax.plot(speed_df['O_time'], speed_df['sudu'], 'o-', color='green', linewidth=2)
        ax.set_xticks(range(24))
        ax.set_xticklabels([f"{h:02d}:00" for h in range(24)])
        
        plt.title('城市道路平均速度变化')
        plt.xlabel('时间')
        plt.ylabel('平均速度 (km/h)')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        return fig
    
    def plot_occupied_taxis(self, occupied_df):
        """绘制载客出租车数量变化图"""
        fig, ax = plt.subplots(figsize=(10, 6)) # 调整为更适合时间序列的尺寸
        
        # 转换时间格式为字符串
        if isinstance(occupied_df['TIME'].iloc[0], datetime):
            time_labels = occupied_df['TIME'].dt.strftime('%H:%M')
        else:
            time_labels = occupied_df['TIME']
        
        # 选择部分数据点以避免过度拥挤
        step = max(1, len(occupied_df) // 48)  # 每半小时一个点
        
        ax.plot(range(len(occupied_df)), occupied_df['number'], color='blue', linewidth=1)
        ax.set_xticks(range(0, len(occupied_df), step))
        ax.set_xticklabels(time_labels[::step], rotation=45)
        
        plt.title('载客出租车数量变化')
        plt.xlabel('时间')
        plt.ylabel('载客出租车数量')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        return fig
    
    def plot_distance_distribution(self, distance_df):
        """绘制出行距离分布图"""
        fig, ax = plt.subplots(figsize=self.default_figsize)
        
        # 设置柱状图
        bar_width = 0.25
        index = range(len(distance_df))
        
        # 绘制三种距离类别的柱状图
        if 'near' in distance_df.columns:
            ax.bar([i - bar_width for i in index], distance_df['near'], 
                  bar_width, label='短途(<4km)', color='green')
        
        if 'middle' in distance_df.columns:
            ax.bar(index, distance_df['middle'], 
                  bar_width, label='中途(4-8km)', color='blue')
        
        if 'far' in distance_df.columns:
            ax.bar([i + bar_width for i in index], distance_df['far'], 
                  bar_width, label='长途(>8km)', color='red')
        
        ax.set_xlabel('日期')
        ax.set_ylabel('订单数量')
        ax.set_title('出行距离分布')
        ax.set_xticks(index)
        ax.set_xticklabels([f"Day {d}" for d in distance_df['day']])
        ax.legend()
        
        plt.tight_layout()
        return fig

    def plot_demand_prediction(self, demand_df):
        """
        绘制乘客需求预测的时间分布图。
        demand_df: 包含预测需求的数据框，至少包括'hour'和'demand'列。
        """
        if demand_df.empty:
            fig, ax = plt.subplots(figsize=self.default_figsize)
            ax.text(0.5, 0.5, '无需求预测数据', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=14, color='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            return fig

        fig, ax = plt.subplots(figsize=self.default_figsize)
        ax.plot(demand_df['time_unit'], demand_df['demand'], 'o-', color='purple', linewidth=2)
        ax.set_xticks(range(24))
        ax.set_xticklabels([f"{h:02d}:00" for h in range(24)])
        
        plt.title('乘客需求预测 (按小时)')
        plt.xlabel('小时')
        plt.ylabel('预测需求量')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        return fig