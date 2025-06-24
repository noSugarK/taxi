import folium
import pandas as pd
import numpy as np
import os
import time, datetime


def generate_order_line_map(od_gdf):
    m = folium.Map(
        location=[22.6, 114],
        zoom_start=10,
        control_scale=True,
        # 使用国内可访问的高德地图瓦片服务
        tiles='http://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=7&x={x}&y={y}&z={z}',
        attr='高德地图'
    )

    start_lat_col = 'O_lat'
    start_lng_col = 'O_lng'
    end_lat_col = 'D_lat'
    end_lng_col = 'D_lng'

    for col in [start_lat_col, start_lng_col, end_lat_col, end_lng_col]:
        if col not in od_gdf.columns:
            raise ValueError(f"缺少列: {col}")

    for i in range(len(od_gdf.head(1000))):
        folium.PolyLine(
            [[od_gdf[start_lat_col].iloc[i], od_gdf[start_lng_col].iloc[i]],
             [od_gdf[end_lat_col].iloc[i], od_gdf[end_lng_col].iloc[i]]],
            color='blue',
            weight=0.5,
            opacity=0.6
        ).add_to(m)

    temp_dir = os.path.abspath("temp_plots")
    os.makedirs(temp_dir, exist_ok=True)
    map_path = os.path.join(temp_dir, "订单线映射.html")
    m.save(map_path)

    print(f"订单线地图已生成: {map_path}")
    return map_path


def generate_sample_point_map(df):
    m = folium.Map(
        location=[22.6, 114],
        zoom_start=10,
        control_scale=True,
        # 使用国内可访问的高德地图瓦片服务
        tiles='http://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=7&x={x}&y={y}&z={z}',
        attr='高德地图'
    )

    points = ([22.4, 113.7], [22.8, 114.3])
    folium.Rectangle(bounds=points, color='#ff7800', fill=False, fill_opacity=0.2).add_to(m)

    lat_col = 'lati'
    lng_col = 'long'

    if lat_col not in df.columns or lng_col not in df.columns:
        raise ValueError(f"缺少列: {lat_col} 或 {lng_col}")

    for lat, lng in zip(df[lat_col].head(10000), df[lng_col].head(10000)):
        folium.CircleMarker(
            [lat, lng],
            radius=1,
            color='yellow',
            fill=True,
            fill_color='yellow',
            fill_opacity=0.6
        ).add_to(m)

    temp_dir = os.path.abspath("temp_plots")
    os.makedirs(temp_dir, exist_ok=True)
    map_path = os.path.join(temp_dir, "采样点映射.html")
    m.save(map_path)

    print(f"采样点地图已生成: {map_path}")
    return map_path