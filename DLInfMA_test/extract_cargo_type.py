import os
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
from shapely.geometry import Point, LineString
import geopandas as gpd
import warnings

warnings.filterwarnings('ignore')


def get_cargo_type():
    wayill_data = pd.read_csv('/Volumes/T7/TDCT/way2endname_DD_unique_renoise.csv')

    # 首先获取所有的停留点
    file_path = '/Volumes/T7/TDCT/cluster_result_stay_point_remove_o_dbscan.csv'
    sp_data = pd.read_csv(file_path)
    # 将未聚类成功的停留点去除
    sp_data = sp_data[sp_data['label'] != -1]

    sr_carge_type = {}

    for cluster_label, sr in tqdm(sp_data.groupby('label')):
        trip_set = set(sr['plan_no'])
        # 运输货物类型（个数）
        cargo_type_set = set()
        for trip in trip_set:
            cargo_type = wayill_data[wayill_data['plan_no'] == trip]['product_name'].values[0]
            cargo_type_set.add(cargo_type)
        sr_carge_type[cluster_label] = cargo_type_set

    return sr_carge_type


# sr_carge_type = get_cargo_type()
# np.save('sr_carge_type.npy', sr_carge_type)

if __name__ == '__main__':
    load_dict = np.load('sr_carge_type.npy', allow_pickle=True)
    print(type(load_dict))
