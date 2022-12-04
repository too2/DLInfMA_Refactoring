"""
Distinguish between detailed address and manual address
"""
import pandas as pd
import numpy as np
import os
from objclass import StayRegion
import utils
from tqdm import tqdm
from read_data import preprocess_way2endname


def Screen(LocationName):
    city_list = ['济南', '青岛', '淄博', '枣庄', '东营', '烟台', '潍坊', '济宁', '泰安', '威海', '日照', '莱芜', '临沂',
                 '德州', '聊城', '滨州', '荷泽']
    Sheng = '山东'
    TempName = LocationName  # TempName存储中间生成地址串，其长度可能会在添加省市等字符后变长
    judge = True  # 用于判断是否是有用地址，True代表是有用地址
    if LocationName.find('省') < 0 and LocationName.find('市') < 0:
        return False
    ShengIndex = LocationName.find(Sheng)  # 返回地址串中'山东'字符串所在下标
    '''
    若地址串中不含'山东'字符串，先检查地址串中有无山东的地级市，若有则补充原字符串为**山东省**市**
    '''
    if ShengIndex < 0:
        for i in range(len(city_list)):
            if LocationName.find(city_list[i]) >= 0:
                TempName = LocationName.replace(city_list[i], Sheng + "省" + city_list[i] + "市")
                break
    if TempName.find(Sheng) < 0:  # 若地址串中没有山东省则直接返回空值
        return False
    ShiIndex = TempName.rfind('市')  # 字符'市'最后一次出现的下标，以下同理
    QuIndex = TempName.rfind('区')
    XianIndex = TempName.rfind('县')
    ZhenIndex = TempName.rfind('镇')
    ShengIndex = TempName.rfind('省')
    '''
    依次检查地址串中有无字符'省','市','区','县',若有,并且这些字符后没有字符，则判定为模糊地址
    '''
    if ShengIndex > 0 and len(TempName) - ShengIndex < 2:
        judge = False
    elif ShiIndex > 0 and len(TempName) - ShiIndex < 2:
        judge = False
    elif QuIndex > 0 and len(TempName) - QuIndex < 2:
        judge = False
    elif XianIndex > 0 and len(TempName) - XianIndex < 2:
        judge = False
    return judge


def get_new_detail_end_point():
    save_path = '/Volumes/T7/TDCT'
    save_filename = 'new_traj_flow.csv'
    traj_data = pd.read_csv(os.path.join(save_path, save_filename), usecols=['plan_no'])
    plan_no_set = set(traj_data['plan_no'])

    way2endname_DD_unique = preprocess_way2endname()
    way2endname_DD_unique_renoise = way2endname_DD_unique[way2endname_DD_unique['plan_no'].isin(plan_no_set)]
    # way2endname_DD_unique_renoise[['plan_no', 'end_point']].to_csv('/Volumes/T7/TDCT/plan_no2end_point.csv',
    #                                                                index=False)

    t_point = pd.read_csv('/Volumes/T7/traj_file/ods_db_sys_t_point.csv')
    t_point[['longitude', 'latitude']] = t_point.apply(lambda x: utils.gcj02_to_wgs84(x.longitude, x.latitude),
                                                       axis=1, result_type="expand")

    end_point_set = set(way2endname_DD_unique_renoise['end_point'])
    end_point_lab_address = t_point[t_point['location_id'].isin(
        end_point_set)][['location_id', 'longitude', 'latitude', 'address']]
    end_point_lab_address.rename(columns={'location_id': 'end_point'}, inplace=True)
    # end_point_lab_address.to_csv('/Volumes/T7/TDCT/end_point_lab_address.csv', index=False)  # 1597->

    detail_end_point_lab_address = end_point_lab_address[end_point_lab_address.apply(
        lambda x: Screen(x.address), axis=1)]  # ->1257
    # detail_end_point_lab_address.to_csv('/Volumes/T7/TDCT/new_detail_end_point.csv', index=False)
    '''
    data format:
    end_point	lng	        lat	        address
    P000016567	118.432815	34.927064	山东省临沂市河东区临沂市沃尔沃路钢材物流城东区494号
    '''
    return end_point_lab_address, detail_end_point_lab_address


# 获取地址库中详细地址，参见 /Users/kxz/JupyterProjects/DASFAA/山东省数据分析IV.ipynb
end_point_lab_address, detail_end_point_lab_address = get_new_detail_end_point()

# 获取候选位置坐标和ID
cluster = pd.read_csv('/Volumes/T7/TDCT/cluster_result_stay_point_remove_o_dbscan.csv')
plan_no2end_point = pd.read_csv('/Volumes/T7/TDCT/plan_no2end_point.csv')
cluster = pd.merge(cluster, plan_no2end_point, on='plan_no')
print(cluster.columns)

sr_dict = {}

for label, sp_data in tqdm(cluster.groupby('label')):
    if label == -1:
        continue
    sr = StayRegion(label)
    sr.cen_lng = np.mean(sp_data['longitude'])
    sr.cen_lat = np.mean(sp_data['latitude'])
    sr.end_point_set = set(sp_data['end_point'])
    sr_dict[label] = sr

end_point_label = []
for index, end_point in tqdm(detail_end_point_lab_address.iterrows()):
    flag = False  # 判断该运输终点是否关联停留热点
    ans_small_1000 = 0
    dist_min = float('inf')
    tmp_label = 0
    tmp_sr = None
    end_point_id = end_point['end_point']
    end_point_address = end_point['address']
    end_point_lng = end_point['longitude']
    end_point_lat = end_point['latitude']
    for label, sr in sr_dict.items():
        if end_point_id not in sr.end_point_set:
            continue
        dist = utils.haversine_distance_loc_points(end_point_lng, end_point_lat, sr.cen_lng, sr.cen_lat)
        if dist < 1000 and end_point_id in set(sr.end_point_set):
            ans_small_1000 += 1
        if dist < dist_min and dist < 1000:
            dist_min = dist
            tmp_label = label
            tmp_sr = sr
            flag = True
    if flag and ans_small_1000 == 1:
        end_point_label.append([end_point_id, end_point_address, tmp_sr.cen_lng, tmp_sr.cen_lat, end_point_lng,
                                end_point_lat, tmp_label])

print('all addresses', len(end_point_lab_address))
print('detail addresses', len(detail_end_point_lab_address))
print('auto tag', len(end_point_label))

# save data
df = pd.DataFrame(data=end_point_label, columns=['end_point', 'location_name', 'longitude', 'latitude', 'tag_longitude',
                                                 'tag_latitude', 'cluster_label'])
# df.to_csv('/Volumes/T7/TDCT/annotation/end_point_lab_final.csv', index=False)

# 获取聚类后的停留点、地址关系
cluster = pd.merge(cluster, end_point_lab_address, on='end_point')
# cluster.to_csv('/Volumes/T7/TDCT/cluster_sp_address.csv', index=False)

# 获取标注文件
all_tag_file = end_point_lab_address
auto_tag_file = detail_end_point_lab_address
auto_tag_end_point_set = set(auto_tag_file['end_point'])
artificial_tag = []
for index, value in all_tag_file.iterrows():
    end_point = value['end_point']
    if end_point in auto_tag_end_point_set:
        continue
    artificial_tag.append(value)
artificial_tag_file = pd.DataFrame(data=artificial_tag, columns=all_tag_file.columns)
# artificial_tag_file.to_csv('/Volumes/T7/TDCT/人工标注数据集/artificial_tag_end_point.csv', index=False)
