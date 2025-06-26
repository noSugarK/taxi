import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import os
import webbrowser
import json
import tempfile
from datetime import datetime

# 导入自定义模块
from data_cleaner import DataCleaner
from data_analyzer import DataAnalyzer
from data_visualizer import DataVisualizer
from prediction_model import PredictionModel
from map_generator import generate_order_line_map, generate_sample_point_map
from dynamic_heatmap import generate_heatmap_data, generate_heatmap_html

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class TaxiGPSAnalyzer:
    def __init__(self):
        self.cleaner = DataCleaner()
        self.analyzer = DataAnalyzer()
        self.visualizer = DataVisualizer()
        # self.predictor = PredictionModel()

    def process_file(self, file, min_long, max_long, min_lati, max_lati, max_speed):
        # 创建临时文件夹存储图表
        temp_dir = "temp_plots"
        os.makedirs(temp_dir, exist_ok=True)

        # 读取CSV文件
        df = self.cleaner.load_data(file.name)
        print(f"成功读取文件，共{len(df)}行数据")

        # 数据清洗
        df_cleaned = self.cleaner.clean_data(df, min_long, max_long, min_lati, max_lati, max_speed)
        print(f"数据清洗完成，保留{len(df_cleaned)}行数据")
        print("df数据列名", list(df_cleaned.columns))

        # 提取OD数据
        od_data = self.analyzer.extract_od_data(df_cleaned)
        print(f"OD数据提取完成，共{len(od_data)}对OD数据")
        print("OD数据的列名:", list(od_data.columns))

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

        # 生成动态热力图数据
        heatmap_data = generate_heatmap_data(df_cleaned)
        print("动态热力图数据生成完成，包含", len(heatmap_data["time_series"]), "个时间点")

        # 生成可视化结果
        gps_plot_path = os.path.join(temp_dir, "gps_plot.png")
        self.visualizer.plot_gps_points(df_cleaned, gps_plot_path)

        hotspots_plot_path = os.path.join(temp_dir, "hotspots_plot.png")
        self.visualizer.plot_hotspots(hotspots, hotspots_plot_path)

        time_plot_path = os.path.join(temp_dir, "time_plot.png")
        self.visualizer.plot_time_distribution(time_dist, time_plot_path)

        speed_plot_path = os.path.join(temp_dir, "speed_plot.png")
        self.visualizer.plot_speed_by_hour(speed_data, speed_plot_path)

        occupied_plot_path = os.path.join(temp_dir, "occupied_plot.png")
        self.visualizer.plot_occupied_taxis(occupied_data, occupied_plot_path)

        distance_plot_path = os.path.join(temp_dir, "distance_plot.png")
        self.visualizer.plot_distance_distribution(distance_data, distance_plot_path)

        # 生成地图
        order_line_map_path = generate_order_line_map(od_data)
        sample_point_map_path = generate_sample_point_map(df_cleaned)

        # 转换为绝对路径
        order_abs_path = os.path.abspath(order_line_map_path)
        point_abs_path = os.path.abspath(sample_point_map_path)

        # 生成分析摘要
        summary = {
            "数据量": len(df),
            "清洗后数据量": len(df_cleaned),
            "OD对数量": len(od_data),
            "热点簇数量": n_clusters,
            "平均行程距离": f"{od_data['OD_Dis_km'].mean():.2f} km",
            "平均行程时间": f"{od_data['OD_TIME_s'].mean() / 60:.2f} 分钟",
            "平均行驶速度": f"{od_data['OD_Dis_km'].sum() / (od_data['OD_TIME_s'].sum() / 3600):.2f} km/h",
        }

        # 确保所有图像都存在
        for path in [
            gps_plot_path, hotspots_plot_path, time_plot_path,
            speed_plot_path, occupied_plot_path, distance_plot_path
        ]:
            if not os.path.exists(path):
                print(f"警告: 文件不存在: {path}")

        return summary, gps_plot_path, hotspots_plot_path, time_plot_path, speed_plot_path, occupied_plot_path, distance_plot_path, order_abs_path, point_abs_path, heatmap_data


def create_interface():  # Gradio
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
                        gps_plot = gr.Image(label="GPS点分布图")

                    with gr.TabItem("热门上客点"):
                        hotspots_plot = gr.Image(label="热门上客点热力图")

                    with gr.TabItem("时间分布"):
                        time_plot = gr.Image(label="乘客打车时间分布")

                    with gr.TabItem("道路速度"):
                        speed_plot = gr.Image(label="城市道路平均速度变化")

                    with gr.TabItem("载客数量"):
                        occupied_plot = gr.Image(label="载客出租车数量变化")

                    with gr.TabItem("出行距离"):
                        distance_plot = gr.Image(label="出行距离分布")

                    with gr.TabItem("订单线映射"):
                        order_line_map_btn = gr.Button("打开订单线映射")

                    with gr.TabItem("采样点映射"):
                        sample_point_map_btn = gr.Button("打开采样点映射")

                    with gr.TabItem("动态热力图"):
                        heatmap_output = gr.HTML(label="动态热力图",min_height=800,max_height=1000)
                        heatmap_data = gr.JSON(visible=False)  # 隐藏的JSON数据组件

        def open_html(path):
            webbrowser.open_new_tab(f'file:///{path.replace(os.sep, "/")}')

        def process_and_display(file, min_long, max_long, min_lati, max_lati, max_speed):
            """处理文件并返回结果，包括热力图数据"""
            result = analyzer.process_file(file, min_long, max_long, min_lati, max_lati, max_speed)
            # 从结果中提取热力图数据
            heatmap_data = result[-1]
            # 生成热力图HTML
            heatmap_html = generate_heatmap_html(heatmap_data)
            # 返回结果（包括热力图HTML）
            return result[:-1] + (heatmap_html,)

        submit_btn.click(
            fn=process_and_display,
            inputs=[file_input, min_long_input, max_long_input, min_lati_input, max_lati_input, max_speed_input],
            outputs=[summary_output, gps_plot, hotspots_plot, time_plot,
                     speed_plot, occupied_plot, distance_plot,
                     order_line_map_btn, sample_point_map_btn, heatmap_output]
        )

        order_line_map_btn.click(
            fn=lambda path: open_html(path),
            inputs=[order_line_map_btn],
            outputs=[]
        )

        sample_point_map_btn.click(
            fn=lambda path: open_html(path),
            inputs=[sample_point_map_btn],
            outputs=[]
        )

    return iface


if __name__ == "__main__":
    iface = create_interface()
    iface.launch()