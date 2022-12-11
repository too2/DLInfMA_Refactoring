import os
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
from shapely.geometry import Point, LineString
import geopandas as gpd
from multiprocessing import Pool
import time
import warnings

warnings.filterwarnings('ignore')


def get_nearest_road(param):
    cluster_label = param[0]
    lng = param[1]
    lat = param[2]
    per_rn_path = param[3]
    g = nx.read_shp(per_rn_path, simplify=True, strict=False)
    min_dist = float('inf')
    road_type = None
    for u, v, data in g.edges(data=True):
        gdfl = gpd.GeoDataFrame(geometry=[LineString((Point(u), Point(v)))], crs="EPSG:4326")
        gdfp = gpd.GeoDataFrame(geometry=[Point(lng, lat)], crs="EPSG:4326")
        utm = gdfl.estimate_utm_crs()
        dist = gdfl.to_crs(utm).distance(gdfp.to_crs(utm)).values[0]
        if dist < min_dist:
            min_dist = dist
            road_type = data['highway']
    return [cluster_label, road_type, min_dist]


def extract_nearest_road_type():
    pass


# nearest_road_type = get_nearest_road()
# nearest_road_type.to_csv('/Volumes/T7/TDCT/nearest_road_type.csv', index=False)

if __name__ == '__main__':
    param_list = []
    # 首先获取所有的停留点
    file_path = './data/cluster_result_stay_point_remove_o_dbscan.csv'  # /Volumes/T7/TDCT/
    sp_data = pd.read_csv(file_path)
    # 将未聚类成功的停留点去除
    sp_data = sp_data[sp_data['label'] != -1]
    sr_data = []
    for cluster_label, sr in sp_data.groupby('label'):
        sr_data.append([cluster_label, np.mean(sr['longitude']), np.mean(sr['latitude'])])
    sr_data = pd.DataFrame(data=sr_data, columns=['cluster_label', 'longitude', 'latitude'])
    rn_path = '../DLInfMA_Refactoring/data/osm_file'  # /Volumes/T7/TDCT/
    path_list = os.listdir(rn_path)
    path_list.sort(key=lambda x: int(x.split('_')[0]))

    for sr_value, f in tqdm(zip(sr_data.itertuples(), path_list)):
        cluster_label = getattr(sr_value, 'cluster_label')
        lng = getattr(sr_value, 'longitude')
        lat = getattr(sr_value, 'latitude')
        per_rn_path = os.path.join(rn_path, f, 'edges.shp')
        param_list.append((cluster_label, lng, lat, per_rn_path))

    print("开始匹配")
    t1 = time.time()
    with Pool() as pool:
        progress_bar = tqdm(total=len(param_list))
        ans = list(tqdm(pool.imap(get_nearest_road, param_list), total=len(param_list)))
    t2 = time.time()
    print("并行执行时间：{}s".format(int(t2 - t1)))

    save_df = pd.DataFrame(data=ans, columns=['cluster_label', 'road_type', 'dist'])
    save_df.to_csv('./sr_road_type.csv', index=False)
    print(save_df)
