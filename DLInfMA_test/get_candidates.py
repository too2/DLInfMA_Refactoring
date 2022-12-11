# 获取每个运单对应的候选位置
import os
import pandas as pd
import numpy as np
from objclass import Address, StayRegion, StayPoint
from tqdm import tqdm
import random
import pickle

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
# 设置随机数种子
setup_seed(20)


def remove_road_sp():
    re_sp_id_set = set()
    file_path = '/Volumes/T7/TDCT/save_mm_traj_point'
    for cluster_file in tqdm(os.listdir(file_path)):
        for sp_file in os.listdir(os.path.join(file_path, cluster_file)):
            if sp_file[0] == '.':
                continue
            sp_traj_path = os.path.join(file_path, cluster_file, sp_file)
            sp_data = pd.read_csv(sp_traj_path)
            candi_pt_error = sp_data.tail(1)['candi_pt_error'].values[0]
            if candi_pt_error < 5:
                re_sp_id_set.add(sp_file.split('.')[0])
    return re_sp_id_set


def apply_sp_data(row):
    return row['plan_no'] + '_' + str(row['start_stay'].replace(' ', '-'))


def get_candi_and_address():
    # 首先获取所有的停留点
    file_path = '/Volumes/T7/TDCT/cluster_result_stay_point_remove_o_dbscan.csv'
    sp_data = pd.read_csv(file_path)
    # 将未聚类成功的停留点去除
    sp_data = sp_data[sp_data['label'] != -1]

    # # 筛除道路上的停留点
    # re_sp_id_set = remove_road_sp()
    # sp_data['sp_id'] = sp_data.apply(lambda x: apply_sp_data(x), axis=1)
    # sp_data = sp_data[~sp_data['sp_id'].isin(re_sp_id_set)]

    # 获得运单到地址的映射
    data_end_point = pd.read_csv('/Volumes/T7/TDCT/plan_no2end_point.csv')
    data = pd.merge(sp_data, data_end_point, on='plan_no', how='left')

    # # 进行鲁棒性实验
    # # 随机选择5%、10%、15%、20%、25%、30%历史行程进行删除
    # re_trip_set = set()
    # for end_point, value in data.groupby('end_point'):
    #     trip_list = list(set(value['plan_no']))
    #     sample_num = int(len(trip_list)*0)
    #     re_trip_list = random.sample(trip_list, sample_num)
    #     re_trip_set.update(re_trip_list)

    # 获取候选位置--停留点关系映射
    address_candi_dict = {}  # key:地址pid, value:候选位置label集合
    candidate_dict = {}  # key:候选位置label, value:候选位置实例
    # 按照聚类簇ID进行处理
    for label, data_candi in tqdm(data.groupby('label')):
        sr_lng = np.mean(data_candi['longitude'])
        sr_lat = np.mean(data_candi['latitude'])
        staypoiobj_list = []
        for index, data_sp in data_candi.iterrows():
            cen_lng = data_sp['longitude']
            cen_lat = data_sp['latitude']
            start_staytime = data_sp['start_stay']
            duration = data_sp['duration']
            plan_no = data_sp['plan_no']
            dri_id = data_sp['dri_id']
            end_point = data_sp['end_point']
            # # 去除部分行程的停留点
            # if plan_no in re_trip_set:
            #     continue
            sp = StayPoint(cen_lng, cen_lat, start_staytime, duration, plan_no, dri_id, end_point)
            staypoiobj_list.append(sp)
            # 获得地址与聚类簇之间的关联
            address_candi_dict.setdefault(end_point, set()).add(label)
        candi = StayRegion(label, sr_lng, sr_lat)
        candi.staypoiobj_list = staypoiobj_list
        candidate_dict[label] = candi

    with open('candidate_dict.pickle', 'wb') as f:
        pickle.dump(candidate_dict, f)
    with open('address_candi_dict.pickle', 'wb') as f:
        pickle.dump(address_candi_dict, f)
    with open('candidate_dict.pickle', 'rb') as f:
        candidate_dict = pickle.load(f)
    with open('address_candi_dict.pickle', 'rb') as f:
        address_candi_dict = pickle.load(f)

    address_dict = {}  # 存放原始地址实例
    end_point_lab_address = pd.read_csv('/Volumes/T7/TDCT/end_point_lab_address.csv')
    # 由于这里去除了一些不符合条件的运输终点，所以需要对address_list进行处理
    tag = pd.read_csv('/Volumes/T7/TDCT/annotation/end_point_lab_final.csv')
    valid_end_point = tag['end_point'].values.tolist()
    for index, (end_point, candi_label_set) in enumerate(address_candi_dict.items()):
        if end_point not in valid_end_point:
            continue
        # lab_lng, lab_lat = \
        # end_point_lab_rizhao[end_point_lab_rizhao['end_point'] == end_point][['longitude', 'latitude']].values.tolist()[0]
        lab_lng, lab_lat = \
        end_point_lab_address[end_point_lab_address['end_point'] == end_point][['lng', 'lat']].values.tolist()[0]
        cluster_label = tag[tag['end_point'] == end_point]['cluster_label'].values.tolist()[0]
        address = Address(end_point, lab_lng, lab_lat, cluster_label)
        for candi_label in candi_label_set:
            # 记录真实位置在候选位置列表中的索引
            if candi_label == cluster_label:
                address.candiates_label_index = len(address.candiates_list)
            candi = candidate_dict[candi_label]
            address.candiates_list.append(candi)
        if address.candiates_label_index is not None:
            address.real_lng = address.candiates_list[address.candiates_label_index].cen_lng
            address.real_lat = address.candiates_list[address.candiates_label_index].cen_lat
        else:
            print('标注位置并不是实际经过的候选位置')
            print(address.end_point)
            continue
        address_dict[end_point] = address
    return candidate_dict, address_dict
