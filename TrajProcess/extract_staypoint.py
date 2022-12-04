import os.path
import utils
import datetime
import pandas as pd
from tqdm import tqdm
from objclass import TrajPoint, StayPoint


def find_first_exceed_max_distance(pt_list, cur_idx, max_distance):
    """
    :param pt_list: 存放轨迹点实体的列表
    :param cur_idx: 当前索引
    :param max_distance: 最大距离
    :return: 找到第一个超过最大距离的点
    """
    cur_pt = pt_list[cur_idx]
    next_idx = cur_idx + 1
    # find all successors whose distance is within MaxStayDist w.r.t. anchor
    while next_idx < len(pt_list):
        next_pt = pt_list[next_idx]
        dist = utils.haversine_distance(cur_pt, next_pt)
        if dist > max_distance:
            break
        next_idx += 1
    return next_idx


def exceed_max_time(pt_list, cur_idx, next_idx, max_stay_time):
    '''
    :param pt_list: 存放轨迹点实体
    :param cur_idx: 当前索引
    :param next_idx: next idx is the first idx that outside the distance threshold
    :param max_stay_time: 最大停留时间
    :return:
    '''
    time_span = (pt_list[next_idx - 1].time - pt_list[cur_idx].time).total_seconds()
    # the time span is larger than maxStayTimeInSecond, a stay point is detected
    return time_span > max_stay_time



class StayPointDetector:
    def __init__(self, max_stay_dist_in_meter, max_stay_time_in_second):
        self.max_distance = max_stay_dist_in_meter
        self.max_stay_time = max_stay_time_in_second

    def detect(self, traj):
        pass


class StayPointClassicDetector(StayPointDetector):
    def __init__(self, max_stay_dist_in_meter, max_stay_time_in_second):
        super(StayPointClassicDetector, self).__init__(max_stay_dist_in_meter, max_stay_time_in_second)

    def detect(self, pt_list):
        sp_list = []
        if len(pt_list) <= 1:
            return sp_list
        cur_idx = 0
        while cur_idx < len(pt_list) - 1:
            next_idx = find_first_exceed_max_distance(pt_list, cur_idx, self.max_distance)
            if exceed_max_time(pt_list, cur_idx, next_idx, self.max_stay_time):
                sp_list.append((pt_list[cur_idx:next_idx], cur_idx))
                cur_idx = next_idx
            else:
                cur_idx += 1
        return sp_list


def write2csv(ans, save_path):
    sp_list = []
    for sp in ans:
        lng = sp.cen_lng
        lat = sp.cen_lat
        start = sp.start_staytime
        duration = sp.duration
        dist = sp.dist
        plan_no = sp.plan_no
        dri_id = sp.dri_id
        position = sp.position
        sp_list.append([plan_no, dri_id, lng, lat, start, duration, position, dist])
    sp_df = pd.DataFrame(data=sp_list,
                         columns=['plan_no', 'dri_id', 'longitude', 'latitude',
                                  'start_stay', 'duration', 'position', 'dist'])
    try:
        sp_df.to_csv(save_path, index=False)
    except Exception as e:
        print(e)


if __name__ == '__main__':
    # 用停留距离和停留时间去判断
    path = '/Volumes/T7/TDCT'
    filename = 'new_traj_flow.csv'
    traj_data = pd.read_csv(os.path.join(path, filename))

    # traj_data = traj_data[traj_data['plan_no'] == 'DD210108001072']

    threshold_dist = 200
    threshold_time = 640  # 640
    sp_detect = StayPointClassicDetector(threshold_dist, threshold_time)
    staypoint_all_vis = []
    for plan_no, plan_no_traj in tqdm(traj_data.groupby('plan_no')):
        traj_list = []
        ans_staypoint = []
        dri_id = plan_no_traj['dri_id'].values[0]
        for index, row in plan_no_traj.iterrows():
            lng = row['longitude']
            lat = row['latitude']
            time = datetime.datetime.strptime(row['time'], '%Y-%m-%d %H:%M:%S')
            dist = row['dist']
            traj_list.append(TrajPoint(plan_no, lng, lat, time, dist))
        traj_segments = sp_detect.detect(traj_list)
        if len(traj_segments) != 0:
            for traj_segment, position in traj_segments:
                duration = traj_segment[-1].time - traj_segment[0].time
                dist = traj_segment[0].dist
                start = traj_segment[0].time
                cen_lng = 0
                cen_lat = 0
                for point in traj_segment:
                    cen_lng += point.longitude
                    cen_lat += point.latitude
                cen_lng /= len(traj_segment)
                cen_lat /= len(traj_segment)
                sp = StayPoint(cen_lng=cen_lng, cen_lat=cen_lat, start_staytime=start, duration=duration,
                               dist=dist, plan_no=plan_no, dri_id=dri_id, position=position, end_point=None)
                staypoint_all_vis.append(sp)
    save_path = '/Volumes/T7/TDCT'
    save_filename = 'staypoint_all.csv'
    write2csv(staypoint_all_vis, os.path.join(save_path, save_filename))


