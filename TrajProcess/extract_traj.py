# Obtain the road network within four kilometers of all stay regions
import pandas as pd
import os
from tqdm import tqdm
from objclass import StayRegion, StayPoint, Traj, TrajPoint
import time
import utils
DISTANCE_THRE = 4000

def write2csv(ans, save_path):
    point_list = []
    plan_no = None
    for point in ans:
        plan_no = point.plan_no
        lng = point.longitude
        lat = point.latitude
        start = point.time
        point_list.append([plan_no, lng, lat, start])
    point_df = pd.DataFrame(data=point_list,
                            columns=['plan_no', 'longitude', 'latitude', 'start_stay'])
    save_path = os.path.join(save_path, plan_no+'.csv')
    try:
        point_df.to_csv(save_path, index=False)
    except Exception as e:
        print(e)


def extract_traj():
    # 获取轨迹数据
    t0 = time.time()
    path = '/Volumes/T7/TDCT'
    filename = 'new_traj_flow.csv'
    traj_data = pd.read_csv(os.path.join(path, filename))
    print(traj_data.info())
    t1 = time.time()
    print(t1-t0, 's')

    traj_dict = {}  # 存放轨迹的列表
    for plan_no, pre_traj_data in tqdm(traj_data.groupby('plan_no')):
        traj = Traj(plan_no)
        trajpoint_list = []
        for index, point in pre_traj_data.iterrows():
            longitude = point['longitude']
            latitude = point['latitude']
            timestamp = point['time']
            trajpoint = TrajPoint(plan_no, longitude, latitude, timestamp)
            trajpoint_list.append(trajpoint)
        traj.traj_point_list = trajpoint_list
        traj_dict[plan_no] = traj

    # 获取聚类停留热点
    data = pd.read_csv('/Volumes/T7/TDCT/cluster_result_stay_point_remove_o_dbscan.csv')
    print(list(data.columns))

    sr_list = []  # 存放sr实体列表
    for label, cluster_data in tqdm(data.groupby('label')):
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
            position = sp_data['position']
            # dist = sp_data['dist']
            sp = StayPoint(lng, lat, start_staytime, duration, plan_no, dri_id, None, position)
            sp_list.append(sp)
        sr.staypoiobj_list = sp_list
        sr.get_center_loc()
        sr_list.append(sr)

    num_empty_sp_subtraj = 0
    # 提取子轨迹段并保存
    for sr in tqdm(sr_list):
        cluster_id = sr.sr_id
        filepath = os.path.join('/Volumes/T7/TDCT/traj_cleaning', str(cluster_id) + '_cluster_id')
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        for sp in sr.staypoiobj_list:
            plan_no = sp.plan_no
            position = sp.position
            traj = traj_dict[plan_no]
            sub_segment = traj.traj_point_list[:position]
            index = 0
            for index, trajpoint in enumerate(sub_segment[::-1]):
                P1_lat, P1_lng, P2_lat, P2_lng = trajpoint.latitude, trajpoint.longitude, sp.cen_lat, sp.cen_lng
                dist = utils.haversine_distance_loc_points(P1_lng, P1_lat, P2_lng, P2_lat)
                if dist >= DISTANCE_THRE:
                    break
            start_index = len(sub_segment)-index
            sub_traj_point_list = sub_segment[start_index:]
            sp.sub_traj_point_list = sub_traj_point_list
            if len(sp.sub_traj_point_list) == 0:
                print(plan_no)
                num_empty_sp_subtraj += 1
                continue
            write2csv(sub_traj_point_list, filepath)
    print(num_empty_sp_subtraj)
    t2 = time.time()
    print(t2-t0, 's')

if __name__ == '__main__':
    extract_traj()
