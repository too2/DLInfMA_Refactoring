# Obtain the road network within five kilometers of all stay regions
import pandas as pd
import osmnx as ox
import os
from tqdm import tqdm
from objclass import StayRegion, StayPoint

def extract_rn():
    data = pd.read_csv('/Volumes/T7/TDCT/cluster_result_stay_point_remove_o_dbscan.csv')
    print(list(data.columns))

    sr_list = []  # 存放sr实体列表
    for label, cluster_data in data.groupby('label'):
        if label == -1:
            continue
        sr_id = label
        sr = StayRegion(sr_id)
        sp_list = []
        for index, sp_data in cluster_data.iterrows():
            plan_no = sp_data['plan_no']
            lng = sp_data['longitude']
            lat = sp_data['latitude']
            start_staytime = sp_data['start_stay']
            duration = sp_data['duration']
            dri_id = sp_data['dri_id']
            dist = sp_data['dist']
            sp = StayPoint(lng, lat, start_staytime, duration, dist, plan_no, dri_id, None)
            sp_list.append(sp)
        sr.staypoiobj_list = sp_list
        sr.get_center_loc()
        sr_list.append(sr)

    for sr in tqdm(sr_list):
        cluster_id = sr.sr_id
        filepath = os.path.join('/Volumes/T7/TDCT/osm_file', str(cluster_id)+'_cluster_id')
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        center_point = (sr.cen_lat, sr.cen_lng)
        try:
            g = ox.graph.graph_from_point(center_point, dist=5000, network_type='drive')
        except:
            g = ox.graph.graph_from_point(center_point, dist=10000, network_type='drive')
            print(cluster_id)
        ox.io.save_graph_shapefile(g, filepath)


