import folium
import pandas as pd
import numpy as np
import os
import time, datetime


def wgs84_to_gcj02(wgs_lng, wgs_lat):
    """
    WGS-84坐标系转换为GCJ-02坐标系（中国国家测绘局制定的地理坐标系统）
    """

    dlat = transform_lat(wgs_lng - 105.0, wgs_lat - 35.0)
    dlng = transform_lng(wgs_lng - 105.0, wgs_lat - 35.0)
    radlat = wgs_lat / 180.0 * np.pi
    magic = np.sin(radlat)
    magic = 1 - 0.006693421622965943 * magic * magic
    sqrtmagic = np.sqrt(magic)
    dlat = (dlat * 180.0) / ((6378245.0 * (1 - 0.006693421622965943)) / (magic * sqrtmagic) * np.pi)
    dlng = (dlng * 180.0) / (6378245.0 / sqrtmagic * np.cos(radlat) * np.pi)
    mg_lat = wgs_lat + dlat
    mg_lng = wgs_lng + dlng
    return mg_lng, mg_lat


def transform_lat(x, y):
    ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * np.sqrt(np.abs(x))
    ret += (20.0 * np.sin(6.0 * x * np.pi) + 20.0 * np.sin(2.0 * x * np.pi)) * 2.0 / 3.0
    ret += (20.0 * np.sin(y * np.pi) + 40.0 * np.sin(y / 3.0 * np.pi)) * 2.0 / 3.0
    ret += (160.0 * np.sin(y / 12.0 * np.pi) + 320 * np.sin(y * np.pi / 30.0)) * 2.0 / 3.0
    return ret


def transform_lng(x, y):
    ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * np.sqrt(np.abs(x))
    ret += (20.0 * np.sin(6.0 * x * np.pi) + 20.0 * np.sin(2.0 * x * np.pi)) * 2.0 / 3.0
    ret += (20.0 * np.sin(x * np.pi) + 40.0 * np.sin(x / 3.0 * np.pi)) * 2.0 / 3.0
    ret += (150.0 * np.sin(x / 12.0 * np.pi) + 300.0 * np.sin(x / 30.0 * np.pi)) * 2.0 / 3.0
    return ret


def convert_coordinates(df, lng_col, lat_col, convert_function=wgs84_to_gcj02):
    """
    对DataFrame中的经纬度列进行坐标转换

    参数:
        df: 包含经纬度的DataFrame
        lng_col: 经度列名
        lat_col: 纬度列名
        convert_function: 坐标转换函数，默认为WGS-84到GCJ-02

    返回:
        转换后的DataFrame
    """
    converted_df = df.copy()
    converted_df[[lng_col, lat_col]] = df.apply(
        lambda row: pd.Series(convert_function(row[lng_col], row[lat_col])),
        axis=1
    )
    return converted_df


def generate_order_line_map(od_gdf, convert=True):
    """
    生成订单线地图

    参数:
        od_gdf: 包含订单起点和终点坐标的GeoDataFrame
        convert: 是否进行坐标转换，默认为True
    """
    # 如果需要，进行坐标转换
    if convert:
        od_gdf = convert_coordinates(od_gdf, 'O_lng', 'O_lat')
        od_gdf = convert_coordinates(od_gdf, 'D_lng', 'D_lat')

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

    # 优化性能：使用pandas的向量化操作替代循环
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


def generate_sample_point_map(df, convert=True):
    """
    生成采样点地图

    参数:
        df: 包含采样点坐标的DataFrame
        convert: 是否进行坐标转换，默认为True
    """
    # 如果需要，进行坐标转换
    if convert:
        df = convert_coordinates(df, 'long', 'lati')

    m = folium.Map(
        location=[22.6, 114],
        zoom_start=10,
        control_scale=True,
        # 使用国内可访问的高德地图瓦片服务
        tiles='http://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=7&x={x}&y={y}&z={z}',
        attr='高德地图'
    )

    lat_col = 'lati'
    lng_col = 'long'

    if lat_col not in df.columns or lng_col not in df.columns:
        raise ValueError(f"缺少列: {lat_col} 或 {lng_col}")

    # 优化性能：使用pandas的向量化操作替代循环
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