import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt

# 导入自定义模块
from data_cleaner import DataCleaner
from data_analyzer import DataAnalyzer
from data_visualizer import DataVisualizer
from prediction_model import PredictionModel

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class TaxiGPSAnalyzer:
    def __init__(self):
        self.cleaner = DataCleaner()
        self.analyzer = DataAnalyzer()
        self.visualizer = DataVisualizer()
        self.predictor = PredictionModel()
    
    def process_file(self, file, min_long, max_long, min_lati, max_lati, max_speed):

        # 读取CSV文件
        df = self.cleaner.load_data(file.name)
        print(f"成功读取文件，共{len(df)}行数据")
        
        # 数据清洗
        df_cleaned = self.cleaner.clean_data(df, min_long, max_long, min_lati, max_lati, max_speed)
        print(f"数据清洗完成，保留{len(df_cleaned)}行数据")
        
        # 提取OD数据
        od_data = self.analyzer.extract_od_data(df_cleaned)
        print(f"OD数据提取完成，共{len(od_data)}对OD数据")
        
        # 热点聚类分析
        hotspots, n_clusters = self.analyzer.cluster_pickup_points(od_data)
        print(f"热点聚类分析完成，发现{n_clusters}个簇")
        
        # 时间分布分析
        time_dist = self.analyzer.analyze_time_distribution(od_data)
        print("时间分布分析完成")

        # 速度分析
        speed_data = self.analyzer.calculate_average_speed(od_data)
        print("速度分析完成")
        
        # 载客数量分析
        occupied_data = self.analyzer.count_occupied_taxis(od_data)
        print("载客数量分析完成")
        
        # 距离分析
        distance_data = self.analyzer.analyze_trip_distance(od_data)
        print("距离分析完成")
        
        # 乘客需求预测 (示例)
        demand_prediction = self.predictor.predict_demand(od_data, 'hourly')
        print("乘客需求预测完成")

        
        eta_prediction = "N/A"
        if not od_data.empty:
            first_od = od_data.iloc[0]
            start_loc = (first_od['O_lng'], first_od['O_lat'])
            end_loc = (first_od['D_lng'], first_od['D_lat'])
            current_time = first_od['O_time'] # 使用订单的起始时间作为当前时间示例
            eta_prediction = self.predictor.predict_eta(start_loc, end_loc, current_time)
            print("ETA预测完成")

        # 生成可视化结果
        gps_plot = self.visualizer.plot_gps_points(df_cleaned)
        hotspots_plot = self.visualizer.plot_hotspots(hotspots)
        time_plot = self.visualizer.plot_time_distribution(time_dist)
        speed_plot = self.visualizer.plot_speed_by_hour(speed_data)
        occupied_plot = self.visualizer.plot_occupied_taxis(occupied_data)
        distance_plot = self.visualizer.plot_distance_distribution(distance_data)
        demand_plot = self.visualizer.plot_demand_prediction(demand_prediction)
        
        # 生成分析摘要
        summary = {
            "数据量": len(df),
            "清洗后数据量": len(df_cleaned),
            "OD对数量": len(od_data),
            "热点簇数量": n_clusters,
            "平均行程距离": f"{od_data['OD_Dis_km'].mean():.2f} km",
            "平均行程时间": f"{od_data['OD_TIME_s'].mean()/60:.2f} 分钟",
            "平均行驶速度": f"{od_data['OD_Dis_km'].sum()/(od_data['OD_TIME_s'].sum()/3600):.2f} km/h",
            "乘客需求预测 (小时)": demand_prediction.to_dict('records') if not demand_prediction.empty else "无数据",
            "首个订单ETA预测": eta_prediction
        }
        
        return summary, gps_plot, hotspots_plot, time_plot, speed_plot, occupied_plot, distance_plot, demand_plot


def create_interface():# Gradio
    analyzer = TaxiGPSAnalyzer()
    
    with gr.Blocks(title="出租车GPS数据分析系统") as iface:
        gr.Markdown("# 出租车GPS数据时空特征提取及可视化系统")
        gr.Markdown("上传包含出租车GPS数据的CSV文件，系统将进行数据清洗、分析及可视化。")
        
        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(label="上传CSV文件")
                
                with gr.Accordion("清洗参数设置", open=False):
                    min_long_input = gr.Number(label="最小经度", value=113.75)
                    max_long_input = gr.Number(label="最大经度", value=114.6)
                    min_lati_input = gr.Number(label="最小纬度", value=22.4)
                    max_lati_input = gr.Number(label="最大纬度", value=22.85)
                    max_speed_input = gr.Number(label="最大速度 (km/h)", value=120)

                submit_btn = gr.Button("开始分析")
                summary_output = gr.JSON(label="分析摘要")
            
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("GPS点分布"):
                        gps_plot = gr.Plot(label="GPS点分布图")
                    
                    with gr.TabItem("热门上客点"):
                        hotspots_plot = gr.Plot(label="热门上客点热力图")
                    
                    with gr.TabItem("时间分布"):
                        time_plot = gr.Plot(label="乘客打车时间分布")
                    
                    with gr.TabItem("道路速度"):
                        speed_plot = gr.Plot(label="城市道路平均速度变化")
                    
                    with gr.TabItem("载客数量"):
                        occupied_plot = gr.Plot(label="载客出租车数量变化")
                    
                    with gr.TabItem("出行距离"):
                        distance_plot = gr.Plot(label="出行距离分布")
                    
                    with gr.TabItem("需求预测"):
                        demand_plot = gr.Plot(label="乘客需求预测")
        
        submit_btn.click(
            fn=analyzer.process_file,
            inputs=[file_input, min_long_input, max_long_input, min_lati_input, max_lati_input, max_speed_input],
            outputs=[summary_output, gps_plot, hotspots_plot, time_plot, speed_plot, occupied_plot, distance_plot, demand_plot]
        )

    return iface

if __name__ == "__main__":
    iface = create_interface()
    iface.launch()