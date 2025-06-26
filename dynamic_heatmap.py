import pandas as pd
import os
import json
import tempfile
import numpy as np
from datetime import datetime
import geopandas as gpd
from dateutil.parser import parse


def generate_heatmap_data(df):
    """生成动态热力图所需的数据格式，时间戳保留到秒"""
    # 验证必要字段存在
    required_columns = ['long', 'lati', 'time']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame必须包含{required_columns}列")

    heatmap_df = df[['long', 'lati', 'time']].copy()

    # 转换时间为时间戳（秒）并添加时间分组
    heatmap_df['timestamp'] = pd.to_datetime(heatmap_df['time']).apply(
        lambda x: int(x.timestamp())  # 保留到秒
    )

    # 按分钟分组（确保时间进度条精确到分钟）
    heatmap_df['time_minute'] = pd.to_datetime(heatmap_df['time']).dt.floor('min')

    # 采样数据点（避免大数据量性能问题）
    if len(heatmap_df) > 10000:
        sampled_dfs = []
        for minute, group in heatmap_df.groupby('time_minute'):
            sampled = group.sample(min(len(group), 200))
            sampled_dfs.append(sampled)
        heatmap_df = pd.concat(sampled_dfs, ignore_index=True)

    # 准备按时间分组的数据
    time_groups = heatmap_df.groupby('time_minute')

    # 生成时间序列和对应的点数据
    time_series = []
    heatmap_data = {}

    for time_minute, group in time_groups:
        timestamp = int(time_minute.timestamp())  # 秒级时间戳
        time_series.append(timestamp)

        points = group.apply(
            lambda row: {
                "point": [row['long'], row['lati']],
                "count": 1  # 每个点代表一个计数
            },
            axis=1
        ).to_list()

        heatmap_data[timestamp] = points

    # 构建完整的热力图数据结构
    result = {
        "time_series": time_series,  # 按顺序排列的时间戳（秒）
        "heatmap_data": heatmap_data,  # 时间戳到点数据的映射
        "min_time": min(time_series) if time_series else 0,
        "max_time": max(time_series) if time_series else 0
    }

    return result


def generate_heatmap_html(heatmap_data, map_bounds=None):
    """生成动态热力图的HTML文件，仅显示当前分钟数据"""
    # 创建临时目录
    temp_dir = "temp_heatmap"
    os.makedirs(temp_dir, exist_ok=True)

    # 保存热力图数据（时间戳到秒）
    data_path = os.path.join(temp_dir, "heatmap_data.json")
    with open(data_path, 'w', encoding='utf-8') as f:
        json.dump(heatmap_data, f, ensure_ascii=False, indent=2)

    # 读取城市边界数据（用于设置地图范围）
    try:
        # 假设城市边界数据位于sz目录下
        sz_path = os.path.join(os.path.dirname(__file__), 'sz', 'sz.shp')
        sz = gpd.read_file(sz_path, encoding='utf-8')

        # 计算城市边界范围
        min_lng, min_lat, max_lng, max_lat = sz.total_bounds
        lng_span = max_lng - min_lng
        lat_span = max_lat - min_lat

        # 创建缓冲区
        lng_buffer = lng_span * 0.1
        lat_buffer = lat_span * 0.1

        map_bounds = {
            "min_lng": min_lng - lng_buffer,
            "max_lng": max_lng + lng_buffer,
            "min_lat": min_lat - lat_buffer,
            "max_lat": max_lat + lat_buffer
        }
        print("成功加载城市边界数据，设置地图范围")
    except Exception as e:
        print(f"加载城市边界数据失败: {e}")
        # 若加载失败，从热力图数据中计算边界
        if map_bounds is None and heatmap_data.get("heatmap_data"):
            first_time_data = next(iter(heatmap_data["heatmap_data"].values()), [])
            if first_time_data:
                coords = [point["point"] for point in first_time_data]
                lngs = [coord[0] for coord in coords]
                lats = [coord[1] for coord in coords]

                min_lng, max_lng = min(lngs), max(lngs)
                min_lat, max_lat = min(lats), max(lats)

                # 创建缓冲区
                lng_buffer = (max_lng - min_lng) * 0.1
                lat_buffer = (max_lat - min_lat) * 0.1

                map_bounds = {
                    "min_lng": min_lng - lng_buffer,
                    "max_lng": max_lng + lng_buffer,
                    "min_lat": min_lat - lat_buffer,
                    "max_lat": max_lat + lat_buffer
                }

    # 计算地图中心点和缩放级别
    if map_bounds:
        center_lng = (map_bounds["min_lng"] + map_bounds["max_lng"]) / 2
        center_lat = (map_bounds["min_lat"] + map_bounds["max_lat"]) / 2

        # 计算合适的缩放级别（基于边界范围）
        lng_span = map_bounds["max_lng"] - map_bounds["min_lng"]
        lat_span = map_bounds["max_lat"] - map_bounds["min_lat"]
        span = max(lng_span, lat_span)

        # 根据范围设置缩放级别
        if span < 0.05:
            zoom = 16  # 适合小范围区域
        elif span < 0.1:
            zoom = 15
        elif span < 0.5:
            zoom = 13
        elif span < 1:
            zoom = 12
        else:
            zoom = 11  # 适合城市范围
    else:
        # 默认设置为深圳中心点
        center_lng, center_lat = 114.0579, 22.5429
        zoom = 11

    # HTML内容（修正字符串模板问题）
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>出租车GPS动态热力图（当前分钟）</title>
        <style>
            html, body {{ height: 800px; margin: 0; padding: 0; }}
            #map-container {{ width: 100%; height: 800px; }}
            .controls {{
                position: absolute; top: 10px; right: 10px; z-index: 1000;
                background: rgba(255,255,255,0.8); padding: 10px; border-radius: 5px;
                box-shadow: 0 0 10px rgba(0,0,0,0.3); font-family: Arial, sans-serif;
            }}
            .control-group {{ margin: 5px 0; display: flex; align-items: center; }}
            .control-label {{ flex: 0 0 80px; font-size: 12px; }}
            input[type="range"] {{ flex: 1; margin-right: 10px; }}
            .btn {{ padding: 5px 10px; margin: 0 5px; cursor: pointer; border: 1px solid #ddd; border-radius: 3px; }}
            .btn:hover {{ background-color: #f0f0f0; }}
            .time-display {{ font-weight: bold; }}
            .time-markers {{ display: flex; justify-content: space-between; margin-top: 5px; font-size: 10px; color: #666; }}
            .time-range {{ font-size: 12px; color: #333; margin-top: 5px; font-weight: bold; }}
        </style>
        <script type="text/javascript" src="http://api.map.baidu.com/api?v=2.0&ak=dVgdjmWfvjaepnifgyPxWLmdCjvNuLij"></script>
        <script type="text/javascript" src="http://api.map.baidu.com/library/Heatmap/2.0/src/Heatmap_min.js"></script>
    </head>
    <body>
        <div id="map-container"></div>
        <div class="controls">
            <div class="control-group">
                <div class="control-label">当前时间:</div>
                <span id="time-display" class="time-display">00:00:00</span>
            </div>
            <div class="time-range">
                显示范围: <span id="time-range-display">00:00 - 00:59</span>
            </div>
            <div class="control-group">
                <div class="control-label">时间进度:</div>
                <input type="range" id="time-slider" min="0" max="1439" value="0">
            </div>
            <div class="time-markers">
                <span>00:00</span>
                <span>06:00</span>
                <span>12:00</span>
                <span>18:00</span>
                <span>23:59</span>
            </div>
            <div class="control-group">
                <div class="control-label">合并半径:</div>
                <input type="range" id="radius-slider" min="10" max="50" value="30">
                <span id="radius-value">30</span>
            </div>
            <div class="control-group">
                <div class="control-label">透明度:</div>
                <input type="range" id="opacity-slider" min="20" max="100" value="70">
                <span id="opacity-value">70%</span>
            </div>
            <div class="control-group">
                <div class="control-label">播放控制:</div>
                <button id="play-btn" class="btn">开始</button>
                <button id="pause-btn" class="btn" disabled>暂停</button>
            </div>
        </div>

        <script>
            // 初始化百度地图（全屏显示）
            var map = new BMap.Map("map-container");
            var centerPoint = new BMap.Point({center_lng}, {center_lat});
            map.centerAndZoom(centerPoint, {zoom});
            map.enableScrollWheelZoom();

            // 动画控制
            var isPlaying = true;
            var animationInterval;
            var currentMinute = 0; // 0-1439 (24*60-1)
            var timeSeries = [];
            var heatmapData = {{}};
            var heatmapOverlay;

            // 获取DOM元素
            const playBtn = document.getElementById('play-btn');
            const pauseBtn = document.getElementById('pause-btn');
            const timeSlider = document.getElementById('time-slider');
            const radiusSlider = document.getElementById('radius-slider');
            const opacitySlider = document.getElementById('opacity-slider');
            const radiusValue = document.getElementById('radius-value');
            const opacityValue = document.getElementById('opacity-value');
            const timeDisplay = document.getElementById('time-display');
            const timeRangeDisplay = document.getElementById('time-range-display');

            // 检查Canvas支持
            function isSupportCanvas() {{
                return !!(document.createElement('canvas').getContext && document.createElement('canvas').getContext('2d'));
            }}
            if (!isSupportCanvas()) {{
                alert('当前浏览器不支持Canvas，无法显示热力图');
                document.getElementById('map-container').innerHTML = '<div style="padding:20px;color:red;">浏览器不支持Canvas</div>';
            }}

            // 分钟数转HH:MM:SS格式
            function formatTime(minutes) {{
                const hours = Math.floor(minutes / 60);
                const mins = minutes % 60;
                return hours.toString().padStart(2, '0') + ':' + mins.toString().padStart(2, '0') + ':00';
            }}

            // 分钟数转时间范围格式（如00:00 - 00:59）
            function formatTimeRange(minutes) {{
                const hours = Math.floor(minutes / 60);
                const mins = minutes % 60;
                const nextMins = (mins + 1) % 60;
                const nextHours = nextMins === 0 ? (hours + 1) % 24 : hours;

                // 使用ES5字符串拼接替代模板字符串，避免Python f-string解析问题
                return hours.toString().padStart(2, '0') + ':' + mins.toString().padStart(2, '0') + ' - ' + 
                       nextHours.toString().padStart(2, '0') + ':' + nextMins.toString().padStart(2, '0');
            }}

            // 构建时间映射表（分钟索引到时间戳）
            function buildTimeMap() {{
                const timeMap = {{}};
                timeSeries.forEach(timestamp => {{
                    const date = new Date(timestamp * 1000); // 秒转毫秒
                    const hour = date.getHours();
                    const minute = date.getMinutes();
                    const key = hour * 60 + minute;
                    timeMap[key] = timestamp;
                }});
                return timeMap;
            }}

            // 加载热力图数据（时间戳到秒）
            fetch('http://127.0.0.1:8000/temp_heatmap/heatmap_data.json')
               .then(response => response.json())
               .then(data => {{
                    timeSeries = data.time_series;
                    heatmapData = data.heatmap_data;

                    if (timeSeries.length === 0) {{
                        document.getElementById('map-container').innerHTML = '<div style="padding:20px;color:orange;">无热力图数据</div>';
                        return;
                    }}

                    const timeMap = buildTimeMap();

                    // 初始化控件
                    timeSlider.max = 1439; // 24小时*60分钟-1
                    timeSlider.value = currentMinute;

                    // 半径滑块事件
                    radiusSlider.addEventListener('input', function() {{
                        radiusValue.textContent = this.value;
                        if (heatmapOverlay) {{
                            heatmapOverlay.setOptions({{radius: parseInt(this.value)}});
                        }}
                    }});

                    // 透明度滑块事件
                    opacitySlider.addEventListener('input', function() {{
                        opacityValue.textContent = this.value + '%';
                        if (heatmapOverlay) {{
                            heatmapOverlay.setOptions({{opacity: this.value / 100}});
                        }}
                    }});

                    // 播放/暂停按钮事件
                    playBtn.addEventListener('click', startAnimation);
                    pauseBtn.addEventListener('click', pauseAnimation);

                    // 显示初始热力图
                    updateDisplay(currentMinute, timeMap);
                    startAnimation();

                    // 时间滑块事件
                    timeSlider.addEventListener('input', function() {{
                        currentMinute = parseInt(this.value);
                        updateDisplay(currentMinute, timeMap);
                        if (!isPlaying) {{
                            clearInterval(animationInterval);
                        }}
                    }});
                }})
               .catch(err => {{
                    console.error('数据加载失败:', err);
                    document.getElementById('map-container').innerHTML = '<div style="padding:20px;color:red;">加载失败: ' + err.message + '</div>';
                }});

            // 更新热力图显示（仅显示当前分钟数据）
            function updateDisplay(minuteIndex, timeMap) {{
                if (minuteIndex < 0 || minuteIndex > 1439) return;

                // 更新时间显示
                timeDisplay.textContent = formatTime(minuteIndex);

                // 更新时间范围显示（如00:00 - 00:59）
                timeRangeDisplay.textContent = formatTimeRange(minuteIndex);

                // 获取当前时间点的时间戳
                const currentTimestamp = timeMap[minuteIndex] || timeSeries[0];

                // 获取当前时间点的数据
                const currentData = heatmapData[currentTimestamp] || [];

                // 转换为百度地图所需格式
                const bmapPoints = currentData.map(point => ({{
                    lng: point.point[0],
                    lat: point.point[1],
                    count: point.count
                }}));

                // 清除旧热力图
                if (heatmapOverlay) {{
                    map.removeOverlay(heatmapOverlay);
                }}

                // 创建新热力图（仅当有数据时）
                if (bmapPoints.length > 0) {{
                    const radius = parseInt(radiusSlider.value);
                    const opacity = parseInt(opacitySlider.value) / 100;

                    // 渐变配置
                    const gradient = {{
                        "0": "rgba(102, 255, 0, 0.3)",
                        "0.5": "rgba(255, 170, 0, 0.6)",
                        "1": "rgba(255, 0, 0, 0.9)"
                    }};

                    heatmapOverlay = new BMapLib.HeatmapOverlay({{
                        radius: radius,
                        opacity: opacity,
                        gradient: gradient
                    }});
                    map.addOverlay(heatmapOverlay);

                    heatmapOverlay.setDataSet({{data: bmapPoints, max: getMaxCount(bmapPoints)}});
                }}
            }}

            // 获取数据中的最大count值
            function getMaxCount(points) {{
                return points.length ? Math.max(...points.map(p => p.count)) : 1;
            }}

            // 开始动画
            function startAnimation() {{
                isPlaying = true;
                playBtn.disabled = true;
                pauseBtn.disabled = false;
                animationInterval = setInterval(() => {{
                    currentMinute = (currentMinute + 1) % 1440;
                    timeSlider.value = currentMinute;
                    const timeMap = buildTimeMap();
                    updateDisplay(currentMinute, timeMap);
                }}, 1000); // 每秒更新一次
            }}

            // 暂停动画
            function pauseAnimation() {{
                isPlaying = false;
                playBtn.disabled = false;
                pauseBtn.disabled = true;
                clearInterval(animationInterval);
            }}

            // 窗口Resize适配
            window.addEventListener('resize', () => map.resize());
        </script>
    </body>
    </html>
    """

    # 保存HTML文件
    html_path = os.path.join(temp_dir, "heatmap.html")
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    # 返回可嵌入的HTML iframe代码
    return f"""
    <div style="width:100%; height:100%; overflow:hidden;">
        <iframe src="http://127.0.0.1:8000/temp_heatmap/heatmap.html" 
                width="100%" 
                height="100%" 
                style="border:none; background:#f0f0f0;">
            您的浏览器不支持iframe，请使用Chrome、Firefox或Edge等现代浏览器。
        </iframe>
    </div>
    """