import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import os
import webbrowser
import json
import tempfile
from datetime import datetime
import pickle
from threading import Lock

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

# 缓存目录
CACHE_DIR = "analysis_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
cache_lock = Lock()  # 用于缓存操作的锁


class TaxiGPSAnalyzer:
    def __init__(self):
        self.cleaner = DataCleaner()
        self.analyzer = DataAnalyzer()
        self.visualizer = DataVisualizer()
        # self.predictor = PredictionModel()

    def process_file(self, file, min_long, max_long, min_lati, max_lati, max_speed, cache_key=None):
        """处理文件并支持增量更新"""
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

        # 保存到缓存
        if cache_key:
            self.save_to_cache(cache_key, {
                "summary": summary,
                "gps_plot_path": gps_plot_path,
                "hotspots_plot_path": hotspots_plot_path,
                "time_plot_path": time_plot_path,
                "speed_plot_path": speed_plot_path,
                "occupied_plot_path": occupied_plot_path,
                "distance_plot_path": distance_plot_path,
                "order_abs_path": order_abs_path,
                "point_abs_path": point_abs_path,
                "heatmap_data": heatmap_data
            })

        return summary, gps_plot_path, hotspots_plot_path, time_plot_path, speed_plot_path, occupied_plot_path, distance_plot_path, order_abs_path, point_abs_path, heatmap_data

    def save_to_cache(self, key, data):
        """保存数据到缓存"""
        with cache_lock:
            cache_path = os.path.join(CACHE_DIR, f"{key}.pkl")
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)

    def load_from_cache(self, key):
        """从缓存加载数据"""
        with cache_lock:
            cache_path = os.path.join(CACHE_DIR, f"{key}.pkl")
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
        return None


def create_interface():  # Gradio
    analyzer = TaxiGPSAnalyzer()
    last_analysis_key = None  # 记录上次分析的缓存键

    with gr.Blocks(title="出租车GPS数据分析系统") as iface:
        gr.Markdown("# 出租车GPS数据时空特征提取及可视化系统")
        gr.Markdown("上传包含出租车GPS数据的CSV文件，系统将进行数据清洗、分析及可视化。")

        # 缓存状态存储
        cache_state = gr.State(None)
        last_analysis_key_state = gr.State(None)
        has_cache_state = gr.State(False)  # 新增状态用于判断是否有缓存

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
                status_output = gr.Text(label="状态")

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
                        # 新增全屏链接按钮
                        fullscreen_link = gr.Button(
                            "全屏观看动态热力图",
                            variant="primary",
                            size="lg"
                        )
                        heatmap_output = gr.HTML(label="动态热力图", min_height=800)

        def check_cache():
            """检查是否有可用缓存"""
            cache_files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.pkl')]
            if cache_files:
                # 选择最新的缓存
                latest_cache = max(cache_files, key=lambda f: os.path.getmtime(os.path.join(CACHE_DIR, f)))
                cache_key = latest_cache[:-4]  # 去除.pkl后缀
                return cache_key, f"找到缓存: {cache_key}", True
            return None, "没有找到分析缓存", False

        def load_cache(cache_key, has_cache):
            """加载缓存数据，确保返回值数量与输出组件匹配"""
            if not has_cache:
                # 返回11个None，与输出组件数量匹配
                return (None,) * 11
            data = analyzer.load_from_cache(cache_key)
            if data:
                heatmap_html = generate_heatmap_html(data["heatmap_data"])
                return (
                    data["summary"],
                    data["gps_plot_path"],
                    data["hotspots_plot_path"],
                    data["time_plot_path"],
                    data["speed_plot_path"],
                    data["occupied_plot_path"],
                    data["distance_plot_path"],
                    data["order_abs_path"],
                    data["point_abs_path"],
                    heatmap_html,
                    f"已加载缓存: {cache_key}"
                )
            # 加载缓存失败时返回11个值
            return (None,) * 10 + ("加载缓存失败",)

        def open_html(path):
            webbrowser.open_new_tab(f'file:///{path.replace(os.sep, "/")}')

        # 获取热力图文件的相对路径
        def get_heatmap_relative_path(heatmap_html):
            if heatmap_html and 'temp_heatmap/heatmap.html' in heatmap_html:
                start = heatmap_html.find('temp_heatmap/heatmap.html')
                if start != -1:
                    end = heatmap_html.find('"', start)
                    if end != -1:
                        return heatmap_html[start:end]
            return "temp_heatmap/heatmap.html"  # 默认相对路径

        # 按钮点击事件处理
        def handle_fullscreen_click(heatmap_html, port=8000):
            heatmap_path = get_heatmap_relative_path(heatmap_html)
            fullscreen_url = f'http://127.0.0.1:{port}/{heatmap_path}'
            webbrowser.open_new_tab(fullscreen_url)

        def process_and_update(file, min_long, max_long, min_lati, max_lati, max_speed, prev_data):
            """处理文件并增量更新结果"""
            if file is None:
                return (None,) * 11  # 没有文件时返回空值
            # 生成缓存键
            cache_key = f"analysis_{datetime.now().strftime('%Y%m%d%H%M%S')}"

            # 处理文件
            result = analyzer.process_file(file, min_long, max_long, min_lati, max_lati, max_speed, cache_key)

            # 提取结果
            summary, gps_plot_path, hotspots_plot_path, time_plot_path, speed_plot_path, occupied_plot_path, distance_plot_path, order_abs_path, point_abs_path, heatmap_data = result

            # 生成热力图HTML
            heatmap_html = generate_heatmap_html(heatmap_data)

            # 构建更新数据
            updated_data = {
                "summary": summary,
                "gps_plot_path": gps_plot_path,
                "hotspots_plot_path": hotspots_plot_path,
                "time_plot_path": time_plot_path,
                "speed_plot_path": speed_plot_path,
                "occupied_plot_path": occupied_plot_path,
                "distance_plot_path": distance_plot_path,
                "order_abs_path": order_abs_path,
                "point_abs_path": point_abs_path,
                "heatmap_html": heatmap_html
            }

            # 只更新有变化的数据，保留其他数据
            if prev_data:
                for key in prev_data:
                    if key not in updated_data or updated_data[key] is None:
                        updated_data[key] = prev_data[key]

            return (updated_data["summary"], updated_data["gps_plot_path"],
                    updated_data["hotspots_plot_path"], updated_data["time_plot_path"],
                    updated_data["speed_plot_path"], updated_data["occupied_plot_path"],
                    updated_data["distance_plot_path"], updated_data["order_abs_path"],
                    updated_data["point_abs_path"], updated_data["heatmap_html"],
                    "分析完成并更新结果")

        # 初始化时检查缓存
        iface.load(
            fn=check_cache,
            outputs=[last_analysis_key_state, status_output, has_cache_state]
        )

        # 加载缓存数据（确保返回值数量正确）
        iface.load(
            fn=load_cache,
            inputs=[last_analysis_key_state, has_cache_state],
            outputs=[summary_output, gps_plot, hotspots_plot, time_plot, speed_plot, occupied_plot, distance_plot,
                     order_line_map_btn, sample_point_map_btn, heatmap_output, status_output]
        )

        # 提交按钮点击事件
        submit_btn.click(
            fn=process_and_update,
            inputs=[file_input, min_long_input, max_long_input, min_lati_input, max_lati_input, max_speed_input,
                    cache_state],
            outputs=[summary_output, gps_plot, hotspots_plot, time_plot, speed_plot, occupied_plot, distance_plot,
                     order_line_map_btn, sample_point_map_btn, heatmap_output, status_output],
        ).then(
            fn=lambda key: key,
            inputs=[last_analysis_key_state],
            outputs=[cache_state]
        )

        # 全屏按钮点击事件
        fullscreen_link.click(
            fn=lambda path: handle_fullscreen_click(path),
            inputs=[fullscreen_link],
            outputs=[],
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