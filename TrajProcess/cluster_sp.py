import pandas as pd
from sklearn.cluster import DBSCAN
from objclass import StayPoint
import utils
from objclass import TrajPoint
from time import time

def calc_avg_loc(coors):
    sum_lng = 0
    sum_lat = 0
    for p in coors:
        sum_lng += p[0]
        sum_lat += p[1]
    return sum_lng / len(coors), sum_lat / len(coors)


def write2csv(ans, labels, save_path):
    sp_list = []
    for sp, label in zip(ans, labels):
        lng = sp.cen_lng
        lat = sp.cen_lat
        start = sp.start_staytime
        duration = sp.duration
        dist = sp.dist
        plan_no = sp.plan_no
        dri_id = sp.dri_id
        position = sp.position
        sp_list.append([plan_no, dri_id, lng, lat, start, duration, dist, position, label])
    sp_df = pd.DataFrame(data=sp_list,
                         columns=['plan_no', 'dri_id', 'longitude', 'latitude', 'start_stay',
                                  'duration', 'dist', 'position', 'label'])
    try:
        sp_df.to_csv(save_path, index=False)
    except Exception as e:
        print(e)


def do_DBSCAN_for_staypoints(spobj_list, eps=5, min_sample=5):
    cenpoi_list = [utils.wgs84_to_mercator(spobj.cen_lng, spobj.cen_lat) for spobj in spobj_list]
    labels = DBSCAN(eps=eps, min_samples=min_sample, metric='euclidean').fit(cenpoi_list).labels_
    for i in range(0, len(labels)):
        spobj_list[i].cluster_id = labels[i]
    return spobj_list, labels


if __name__ == '__main__':
    t0 = time()
    # 对停留点进行聚类
    sp_data_remove_od = pd.read_csv('/Volumes/T7/TDCT/staypoint_all.csv')
    spobj_list = []
    for index, row in sp_data_remove_od.iterrows():
        # 去掉起点10公里范围内，终点3公里范围内的停留点
        # 起点：日照钢厂（119.35426,35.15754）；终点：泰安钢材市场（117.06392,36.07603）
        start_point = TrajPoint(None, 119.35426, 35.15754, None)
        # end_point = TrajPoint(None, None, 117.06392, 36.07603, None)
        curr_point = TrajPoint(None, row['longitude'], row['latitude'], None)
        dist_start = utils.haversine_distance(start_point, curr_point)
        # dist_end = utils.haversine_distance(end_point, curr_point)
        if dist_start < 3000:  # or dist_end < 3000:
            continue
        spobj_list.append(StayPoint(plan_no=row['plan_no'], dri_id=row['dri_id'], cen_lng=row['longitude'],
                                    cen_lat=row['latitude'], start_staytime=row['start_stay'],
                                    duration=row['duration'], dist=row['dist'], end_point=None, position=row['position']))
    miniclus, labels = do_DBSCAN_for_staypoints(spobj_list, eps=50, min_sample=5)
    write2csv(miniclus, labels, '/Volumes/T7/TDCT/cluster_result_stay_point_remove_o_dbscan.csv')
    t1 = time()
    print(t1-t0, 's')

