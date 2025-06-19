import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import matplotlib
from datetime import datetime
import os

# 数据可视化模块
class DataVisualizer:
    def __init__(self):
        self.default_figsize = (8, 6) # 设置默认图片大小
    
    def plot_gps_points(self, df):
        """绘制GPS点分布图（带行政区划边界）"""
        # 读取深圳市行政区划边界数据
        sz_path = os.path.join(os.path.dirname(__file__), 'sz', 'sz.shp')
        sz = gpd.read_file(sz_path, encoding='utf8')
        
        # 创建图形和坐标轴
        fig = plt.figure(figsize=(10, 6), dpi=100)
        ax = plt.subplot(111)
        
        # 绘制行政区划边界
        sz.plot(ax=ax, edgecolor=(0,0,0,1), facecolor=(0,0,0,0.05), linewidths=0.5)
        
        # 绘制GPS点
        scatter = ax.scatter(df['long'], df['lati'], c=df['status'], 
                           cmap='coolwarm', alpha=0.6, marker='o', s=10)
        
        # 添加颜色条
        cax = plt.axes([0.15, 0.33, 0.02, 0.3])
        cbar = plt.colorbar(scatter, cax=cax)
        cbar.set_label('状态 (0=空车, 1=载客)')

        """
        plt.title('出租车GPS点分布')
        plt.xlabel('经度')
        plt.ylabel('纬度')
        
        # 设置显示范围
        plt.xlim(113.7, 114.3)
        plt.ylim(22.4, 22.8)
        """

        fig.suptitle('热门上客点分布', fontsize=16, y=0.95)  # y 控制标题垂直位置
        fig.supxlabel('经度')
        fig.supylabel('纬度')
        # 设置显示范围
        ax.set_xlim(113.7, 114.65)
        ax.set_ylim(22.4, 22.88)
        # 隐藏坐标轴
        plt.axis('off')
        
        return fig
    
    def plot_hotspots(self, hotspots_df):
        """绘制热门上客点热力图（带行政区划边界）"""
        # 读取深圳市行政区划边界数据
        sz_path = os.path.join(os.path.dirname(__file__), 'sz', 'sz.shp')
        sz = gpd.read_file(sz_path, encoding='utf8')
        
        # 创建图形和坐标轴
        fig = plt.figure(figsize=(10, 6), dpi=100)
        ax = plt.subplot(111)
        
        # 绘制行政区划边界
        sz.plot(ax=ax, edgecolor=(0,0,0,1), facecolor=(0,0,0,0.05), linewidths=0.5)
        
        # 设置颜色映射的最大值为数据的99%分位数
        vmax = hotspots_df['count'].quantile(0.99)
        
        # 创建颜色映射和归一化器
        cmap_obj = plt.cm.get_cmap('hot_r')
        norm = matplotlib.colors.Normalize(vmin=0, vmax=vmax)
        
        # 绘制热点散点图
        scatter = ax.scatter(hotspots_df['lng'], hotspots_df['lat'], 
                           s=hotspots_df['count']*3, # 点大小与热力值成正比
                           alpha=0.7, 
                           c=hotspots_df['count'], 
                           cmap=cmap_obj, norm=norm)
        
        # 添加颜色条
        cax = plt.axes([0.15, 0.33, 0.02, 0.3])
        cbar = plt.colorbar(scatter, cax=cax)
        cbar.set_label('上客次数')

        """
        plt.title('热门上客点分布')
        plt.xlabel('经度')
        plt.ylabel('纬度')
        
        # 设置显示范围
        plt.xlim(113.7, 114.3)
        plt.ylim(22.4, 22.8)
        """
        fig.suptitle('热门上客点分布', fontsize=16, y=0.95)  # y 控制标题垂直位置

        # 设置显示范围
        ax.set_xlim(113.7, 114.65)
        ax.set_ylim(22.4, 22.88)
        # 隐藏坐标轴
        plt.axis('off')
        
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
        
    def plot_with_district_boundary(self, data_df, plot_type='scatter', column=None, title='深圳市区域数据分布', cmap='plasma', alpha=0.7, s=10, vmax=None):
        """
        在深圳市行政区划边界上绘制数据分布图
        
        参数:
        data_df: 包含经纬度数据的DataFrame，必须包含'long'/'lng'和'lati'/'lat'列
        plot_type: 绘图类型，可选'scatter'(散点图)或'heatmap'(热力图)
        column: 用于着色的列名，如果为None则使用统一颜色
        title: 图表标题
        cmap: 颜色映射名称
        alpha: 透明度
        s: 点大小
        vmax: 颜色映射的最大值，默认为数据的99%分位数
        
        返回:
        matplotlib图形对象
        """
        # 读取深圳市行政区划边界数据
        sz_path = os.path.join(os.path.dirname(__file__), 'sz', 'sz.shp')
        sz = gpd.read_file(sz_path, encoding='utf8')
        
        # 创建图形和坐标轴
        fig = plt.figure(figsize=(10, 8), dpi=100)
        ax = plt.subplot(111)
        
        # 设置边界范围
        bounds = [113.7, 22.42, 114.3, 22.8]
        
        # 绘制行政区划边界
        sz.plot(ax=ax, edgecolor=(0,0,0,1), facecolor=(0,0,0,0.05), linewidths=0.5)
        
        # 确保数据列名一致
        if 'long' in data_df.columns and 'lng' not in data_df.columns:
            data_df = data_df.rename(columns={'long': 'lng'})
        if 'lati' in data_df.columns and 'lat' not in data_df.columns:
            data_df = data_df.rename(columns={'lati': 'lat'})
        
        # 根据绘图类型选择不同的绘制方法
        if plot_type == 'scatter':
            # 如果提供了用于着色的列
            if column is not None:
                # 设置颜色映射的最大值为数据的99%分位数（如果未指定）
                if vmax is None:
                    vmax = data_df[column].quantile(0.99)
                
                # 创建颜色映射和归一化器
                cmap_obj = plt.cm.get_cmap(cmap)
                norm = matplotlib.colors.Normalize(vmin=0, vmax=vmax)
                
                # 绘制散点图
                scatter = ax.scatter(data_df['lng'], data_df['lat'], 
                                   s=s, alpha=alpha, 
                                   c=data_df[column], 
                                   cmap=cmap_obj, norm=norm)
                
                # 添加颜色条
                cax = plt.axes([0.15, 0.33, 0.02, 0.3])
                cbar = plt.colorbar(scatter, cax=cax)
                cbar.set_label(column)
            else:
                # 不使用颜色映射的简单散点图
                ax.scatter(data_df['lng'], data_df['lat'], s=s, alpha=alpha, color='blue')
        
        elif plot_type == 'heatmap':
            # 确保数据中有count列或使用指定的列
            count_col = column if column is not None else 'count'
            if count_col not in data_df.columns:
                raise ValueError(f"数据中缺少'{count_col}'列，无法绘制热力图")