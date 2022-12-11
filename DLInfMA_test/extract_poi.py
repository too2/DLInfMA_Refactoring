import numpy as np
import pandas as pd
import utils_poi
from requests_html import HTMLSession
import json
from tqdm import tqdm


def request_sr_poi(radius: int, data, keys: list):
    flag = True
    ans = []
    radius_list = [radius]
    keys_index = 0
    for index, center_point in tqdm(data.iterrows()):
        if index <= 2135:
            continue
        if not flag:
            break
        for num in radius_list:
            PoiTypes = ['010100', '050000', '170000']  # , '030000', '180300'
            key = keys[keys_index]
            tmp_ans = []
            for PoiType in PoiTypes:
                params = {
                    "key": key,
                    "location": [str(round(getattr(center_point, 'longitude'), 6)) + ',' + \
                                 str(round(getattr(center_point, 'latitude'), 6))],
                    "types": PoiType,
                    "radius": num,
                    "output": "json",
                }
                url = 'https://restapi.amap.com/v3/place/around'
                session = HTMLSession()
                rq = session.get(url, params=params)
                result = json.loads(rq.html.html)
                if result['status'] == '0':
                    flag = False
                    break
                # 控制时间反爬虫
                # time.sleep(random.randint(3, 4))
                total_count = result['count']
                tmp_ans.append(total_count)
            if not flag:
                break
            cluster_id = str(int(getattr(center_point, 'cluster_label')))
            tmp_ans.extend([cluster_id])
            ans.append(tmp_ans)
    ans = pd.DataFrame(data=ans, columns=['汽车服务_加油站_500', '餐饮服务_500', '公司企业_500', 'cluster_label'])
    return ans

def request_end_point_poi(radius: int, data, keys: list):
    flag = True
    ans = []
    radius_list = [radius]
    keys_index = 0
    for index, center_point in tqdm(data.iterrows()):
        if index < 1295:
            continue
        if not flag:
            break
        for num in radius_list:
            PoiTypes = ['010100', '050000', '170000']  # , '030000', '180300'
            key = keys[keys_index]
            tmp_ans = []
            for PoiType in PoiTypes:
                params = {
                    "key": key,
                    "location": [str(round(getattr(center_point, 'lng'), 6)) + ',' + \
                                 str(round(getattr(center_point, 'lat'), 6))],
                    "types": PoiType,
                    "radius": num,
                    "output": "json",
                }
                url = 'https://restapi.amap.com/v3/place/around'
                session = HTMLSession()
                rq = session.get(url, params=params)
                result = json.loads(rq.html.html)
                if result['status'] == '0':
                    flag = False
                    break
                # 控制时间反爬虫
                # time.sleep(random.randint(3, 4))
                total_count = result['count']
                tmp_ans.append(total_count)
            if not flag:
                break
            cluster_id = getattr(center_point, 'end_point')
            tmp_ans.extend([cluster_id])
            ans.append(tmp_ans)
    ans = pd.DataFrame(data=ans, columns=['汽车服务_加油站_500', '餐饮服务_500', '公司企业_500', 'end_point'])
    return ans

def get_sr_poi():
    # 首先获取所有的停留点
    file_path = '/Volumes/T7/TDCT/cluster_result_stay_point_remove_o_dbscan.csv'
    sp_data = pd.read_csv(file_path)
    # 将未聚类成功的停留点去除
    sp_data = sp_data[sp_data['label'] != -1]
    # 查找每个停留点对应的POI
    sp_data[['longitude', 'latitude']] = sp_data.apply(utils_poi.wgs84togaode_arr, axis=1,
                                                       args=('longitude', 'latitude'), result_type="expand")
    sr_data = []
    for cluster_label, sr in sp_data.groupby('label'):
        sr_data.append([cluster_label, np.mean(sr['longitude']), np.mean(sr['latitude'])])
    sr_data = pd.DataFrame(data=sr_data, columns=['cluster_label', 'longitude', 'latitude'])
    print(len(sr_data))
    # 高德接口
    keys = ['ce12777650a631b309cdca4ea3ab1090']
    data = sr_data[['longitude', 'latitude', 'cluster_label']]
    radius = 500
    PoiTypes = ['010100', '050000', '170000']  # '030000', '180300'
    names = ['汽车服务_加油站', '餐饮服务', '公司企业']  # '汽车维修', '道路附属设施_服务区'
    radius_all = [500]
    columns = {}
    for r in radius_all:
        for name, PoiType in zip(names, PoiTypes):
            columns[PoiType + '_' + str(r)] = name + '_' + str(r)
    ans = request_sr_poi(radius, data, keys)
    ans.rename(columns=columns, inplace=True)
    print(ans.head())
    ans.to_csv('/Volumes/T7/TDCT/candi_poi.csv', index=False)


def get_end_point_poi():
    end_point_data = pd.read_csv('/Volumes/T7/TDCT/end_point_lab_address.csv')
    print(len(end_point_data))
    # 高德接口
    keys = ['6a4d78871c3aac0f548c0bc2e4784546']
    radius = 500
    PoiTypes = ['010100', '050000', '170000']  # '030000', '180300'
    names = ['汽车服务_加油站', '餐饮服务', '公司企业']  # '汽车维修', '道路附属设施_服务区'
    radius_all = [500]
    columns = {}
    for r in radius_all:
        for name, PoiType in zip(names, PoiTypes):
            columns[PoiType + '_' + str(r)] = name + '_' + str(r)
    ans = request_end_point_poi(radius, end_point_data, keys)
    ans.rename(columns=columns, inplace=True)
    print(ans.head())
    ans.to_csv('/Volumes/T7/TDCT/candi_poi.csv', index=False)

def extract_sr_poi():
    sr_poi_data = pd.read_excel('/Volumes/T7/TDCT/爬取POI.xlsx')
    return sr_poi_data

def extract_end_point_poi():
    end_point_poi_data = pd.read_excel('/Volumes/T7/TDCT/爬取POI_end_point.xlsx')
    return end_point_poi_data


if __name__ == '__main__':
    get_end_point_poi()
