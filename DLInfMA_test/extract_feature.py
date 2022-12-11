import get_candidates
from tqdm import tqdm
import pandas as pd
import numpy as np
import datetime
from collections import defaultdict
from shapely.geometry import Polygon
import utils_poi
from extract_poi import extract_sr_poi, extract_end_point_poi
from extract_road_type import extract_nearest_road_type
from objclass import Traj


def get_sr_feature(candidate_dict):
    # 这里是要处理每一个候选位置实例
    # behaviour feature
    # 提取POI文件
    poi = extract_sr_poi()
    for candi_label, candiobj in tqdm(candidate_dict.items()):
        # 候选位置对应行程总数
        trip_set = set()
        # 开始停留时段
        time_distribution = [0] * 24
        # 停留时长
        trip_duration = []
        # 停留频次
        trip_stop = defaultdict(int)
        for spobj in candiobj.staypoiobj_list:
            # 候选位置对应行程总数
            trip_set.add(spobj.plan_no)
            # 开始停留时段
            timestamp = datetime.datetime.strptime(spobj.start_staytime, '%Y-%m-%d %H:%M:%S').hour  # 获得小时
            time_distribution[timestamp] += 1
            # 停留时长
            duration = pd.to_timedelta(spobj.duration).total_seconds() // 60
            trip_duration.append(duration)
            # 停留频次
            plan_no = spobj.plan_no
            trip_stop[plan_no] += 1
        candiobj.trip_set = trip_set
        candiobj.time_distribution = time_distribution
        stop_fre = trip_stop.values()
        # 对停留频次分桶
        stop_fre_bin = defaultdict(int)
        for i in stop_fre:
            if i == 1:
                stop_fre_bin[i] += 1
            elif i == 2:
                stop_fre_bin[i] += 1
            elif i == 3:
                stop_fre_bin[i] += 1
            else:
                stop_fre_bin[4] += 1
        candiobj.stop_fre = [stop_fre_bin[i] for i in range(1, 5)]
        # 对停留时长分桶
        stop_duration_bin = [0 for _ in range(5)]
        for stop_duration in trip_duration:
            if stop_duration / 60 < 10:
                stop_duration_bin[0] += 1
            elif stop_duration / 60 < 15:
                stop_duration_bin[1] += 1
            elif stop_duration / 60 < 30:
                stop_duration_bin[2] += 1
            elif stop_duration / 60 < 60:
                stop_duration_bin[3] += 1
            else:
                stop_duration_bin[4] += 1
        candiobj.avg_duration = stop_duration_bin

        # 运输货物类型（个数）
        cargo_type_dict = np.load('sr_carge_type.npy', allow_pickle=True).item()
        for cluster_label, cargo_type_set in cargo_type_dict.items():
            if candi_label == cluster_label:
                candiobj.cargo_type_set = cargo_type_set
                break

        # 区域特征（邻近道路类型、邻近POI类型、停留热点面积）
        # 停留热点面积
        candiobj.convex_hull = Polygon([(i.cen_lng, i.cen_lng) for i in candiobj.staypoiobj_list])
        candiobj.area = candiobj.convex_hull.area

        # 邻近POI类型
        for index, value in poi.iterrows():
            if candi_label == value['cluster_label']:
                candiobj.poi_type = [value['汽车服务_加油站_500'], value['餐饮服务_500'], value['公司企业_500']]
                break

        # 邻近道路类型
        sr_road_type = pd.read_csv('/Volumes/T7/TDCT/sr_road_type.csv')
        for index, value in sr_road_type.iterrows():
            if candi_label == value['cluster_label']:
                candiobj.nearRoad_type = value['road_type']
                break


def get_address_feature(address_dict):
    # 地址特征（地址对应的行程数量，地址的POI类别）
    for end_point, addressobj in tqdm(address_dict.items()):
        # 地址对应的行程数量
        trip_set = set()
        for candiobj in addressobj.candiates_list:
            for spobj in candiobj.staypoiobj_list:
                trip_set.add(spobj.plan_no)
        addressobj.trip_set = trip_set
        # 地理编码位置的POI类别
        poi = extract_end_point_poi()
        for index, value in poi.iterrows():
            if end_point == value['end_point']:
                addressobj.poi_type = [value['汽车服务_加油站_500'], value['餐饮服务_500'], value['公司企业_500']]
                break


# Feature Extraction
# Matching Features (Trip_coverage)
def extract_feature():
    candidate_dict, address_dict = get_candidates.get_candi_and_address()

    get_sr_feature(candidate_dict)
    get_address_feature(address_dict)

    return_plan_dict = np.load('return_plan_dict.npy', allow_pickle=True).item()

    # 处理候选位置与地址的关联特征
    for key, addressobj in tqdm(address_dict.items()):
        p1_lng, p1_lat = addressobj.lab_lng, addressobj.lab_lat
        end_point = addressobj.end_point
        for candiobj in addressobj.candiates_list:
            # 候选位置距离终点距离
            p2_lng, p2_lat = candiobj.cen_lng, candiobj.cen_lat
            dist = utils_poi.geo_distance(p1_lat, p1_lng, p2_lat, p2_lng)
            candiobj.dist_end[end_point] = dist
            # 候选位置行程覆盖率
            coverage = len(addressobj.trip_set.intersection(candiobj.trip_set)) / len(addressobj.trip_set)
            candiobj.trip_converage[end_point] = coverage

        for plan_no in addressobj.trip_set:
            traj = Traj(plan_no)
            return_loc = return_plan_dict[plan_no]
            traj.return_loc = return_loc
            addressobj.trip_list.append(traj)

    return address_dict
