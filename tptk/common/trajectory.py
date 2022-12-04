import os.path
from datetime import datetime, timedelta
from common.spatial_func import distance, SPoint
from common.mbr import MBR
from map_matching.candidate_point import CandidatePoint
from common.spatial_func import cal_loc_along_line
import pandas as pd


class STPoint(SPoint):
    def __init__(self, lat, lng, time, data=None):
        super(STPoint, self).__init__(lat, lng)
        self.time = time
        self.data = data

    def __str__(self):
        return '({}, {}, {}, {})'.format(self.time.strftime('%Y/%m/%d %H:%M:%S'), self.lat, self.lng, self.data)


class Trajectory:
    def __init__(self, oid, pid, pt_list):
        self.oid = pid
        self.pid = pid
        self.pt_list = pt_list

    def get_duration(self):
        return (self.pt_list[-1].time - self.pt_list[0].time).total_seconds()

    def get_length(self):
        if len(self.pt_list) <= 1:
            return 0.0
        else:
            dist = 0.0
            pre_pt = None
            for pt in self.pt_list:
                if pre_pt is None:
                    pre_pt = pt
                else:
                    tmp_dist = distance(pre_pt, pt)
                    dist += tmp_dist
                    pre_pt = pt
            return dist

    def get_time_interval(self):
        point_time_interval = []
        for pre, cur in zip(self.pt_list[:-1], self.pt_list[1:]):
            point_time_interval.append((cur.time - pre.time).total_seconds())
        return sum(point_time_interval) / len(point_time_interval)

    def get_distance_interval(self):
        point_dist_interval = []
        for pre, cur in zip(self.pt_list[:-1], self.pt_list[1:]):
            point_dist_interval.append(distance(pre, cur))
        return sum(point_dist_interval) / len(point_dist_interval)

    def get_mbr(self):
        return MBR.cal_mbr(self.pt_list)

    def get_start_time(self):
        return self.pt_list[0].time

    def get_end_time(self):
        return self.pt_list[-1].time

    def get_mid_time(self):
        return self.pt_list[0].time + (self.pt_list[-1].time - self.pt_list[0].time) / 2.0

    def get_centroid(self):
        mean_lat = 0.0
        mean_lng = 0.0
        for pt in self.pt_list:
            mean_lat += pt.lat
            mean_lng += pt.lng
        mean_lat /= len(self.pt_list)
        mean_lng /= len(self.pt_list)
        return SPoint(mean_lat, mean_lng)

    def query_trajectory_by_temporal_range(self, start_time, end_time):
        # start_time <= pt.time < end_time
        traj_start_time = self.get_start_time()
        traj_end_time = self.get_end_time()
        if start_time > traj_end_time:
            return None
        if end_time <= traj_start_time:
            return None
        st = max(traj_start_time, start_time)
        et = min(traj_end_time + timedelta(seconds=1), end_time)
        start_idx = self.binary_search_idx(st)  # pt_list[start_idx].time <= st < pt_list[start_idx+1].time
        if self.pt_list[start_idx].time < st:
            # then the start_idx is out of the range, we need to increase it
            start_idx += 1
        end_idx = self.binary_search_idx(et)  # pt_list[end_idx].time <= et < pt_list[end_idx+1].time
        if self.pt_list[end_idx].time < et:
            # then the end_idx is acceptable
            end_idx += 1
        sub_pt_list = self.pt_list[start_idx:end_idx]
        return Trajectory(self.pid, sub_pt_list)

    def binary_search_idx(self, time):
        # self.pt_list[idx].time <= time < self.pt_list[idx+1].time
        # if time < self.pt_list[0].time, return -1
        # if time >= self.pt_list[len(self.pt_list)-1].time, return len(self.pt_list)-1
        nb_pts = len(self.pt_list)
        if time < self.pt_list[0].time:
            return -1
        if time >= self.pt_list[-1].time:
            return nb_pts - 1
        # the time is in the middle
        left_idx = 0
        right_idx = nb_pts - 1
        while left_idx <= right_idx:
            mid_idx = int((left_idx + right_idx) / 2)
            if mid_idx < nb_pts - 1 and self.pt_list[mid_idx].time <= time < self.pt_list[mid_idx + 1].time:
                return mid_idx
            elif self.pt_list[mid_idx].time < time:
                left_idx = mid_idx + 1
            else:
                right_idx = mid_idx - 1

    def query_location_by_timestamp(self, time):
        idx = self.binary_search_idx(time)
        if idx == -1 or idx == len(self.pt_list) - 1:
            return None
        if self.pt_list[idx].time == time or (self.pt_list[idx+1].time - self.pt_list[idx].time).total_seconds() == 0:
            return SPoint(self.pt_list[idx].lat, self.pt_list[idx].lng)
        else:
            # interpolate location
            dist_ab = distance(self.pt_list[idx], self.pt_list[idx+1])
            if dist_ab == 0:
                return SPoint(self.pt_list[idx].lat, self.pt_list[idx].lng)
            dist_traveled = dist_ab * (time - self.pt_list[idx].time).total_seconds() / \
                            (self.pt_list[idx+1].time - self.pt_list[idx].time).total_seconds()
            return cal_loc_along_line(self.pt_list[idx], self.pt_list[idx+1], dist_traveled / dist_ab)

    def to_wkt(self):
        wkt = 'LINESTRING ('
        for pt in self.pt_list:
            wkt += '{} {}, '.format(pt.lng, pt.lat)
        wkt = wkt[:-2] + ')'
        return wkt

    def __hash__(self):
        return hash(self.pid + '_' + self.pt_list[0].time.strftime('%Y%m%d%H%M%S') + '_' +
                    self.pt_list[-1].time.strftime('%Y%m%d%H%M%S'))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __repr__(self):
        return f'Trajectory(pid={self.pid})'


def get_tid(oid, pt_list):
    return oid + '_' + pt_list[0].time.strftime('%Y%m%d%H%M%S') + '_' + pt_list[-1].time.strftime('%Y%m%d%H%M%S')


def traj_point_dist(traj, pt, method='centroid'):
    if method == 'nearest':
        dists = []
        for t_pt in traj.pt_list:
            dists.append(distance(t_pt, pt))
        return min(dists)
    elif method == 'centroid':
        return distance(pt, traj.get_centroid())


def parse_traj_file(input_path, traj_type='raw', extra_fields=None):
    assert traj_type in ['raw', 'mm'], 'only `raw` or `mm` is supported'

    time_format = '%Y-%m-%d %H:%M:%S'
    df = pd.read_csv(input_path)
    pid = df['plan_no'].values[0]
    oid = pid
    trajs = []
    pt_list = []
    for index, value in df.iterrows():
        lat = float(value['latitude'])
        lng = float(value['longitude'])
        pt = STPoint(lat, lng, datetime.strptime(value['start_stay'], time_format), None)
        pt_list.append(pt)
    if len(pt_list) != 0:
        traj = Trajectory(oid, pid, pt_list)
        trajs.append(traj)
    return trajs


def store_traj_file(trajs, target_path, target_path_path, filename, traj_type='raw'):
    assert traj_type in ['raw', 'mm'], 'only `raw` or `mm` is supported'

    time_format = '%Y-%m-%d %H:%M:%S'
    data = []
    for traj, path in trajs:
        pt_list = traj.pt_list
        for pt in pt_list:
            candi_pt = pt.data['candi_pt']
            if candi_pt is not None:
                data.append([pt.time.strftime(time_format), pt.lat, pt.lng, candi_pt.eid, candi_pt.lat, candi_pt.lng,
                            candi_pt.error, candi_pt.offset])
            else:
                data.append([pt.time.strftime(time_format), pt.lat, pt.lng, None, None, None, None, None])
        df = pd.DataFrame(data=data, columns=['timestamp', 'latitude', 'longitude', 'eid', 'candi_pt_lat',
                                              'candi_pt_lon', 'candi_pt_error', 'candi_pt_offset'])
        df.to_csv(target_path, index=False)


        for index, p in enumerate(path):
            p_data = []
            for p_entity in p.path_entities:
                p_data.append([p_entity.eid, str(p_entity.enter_time), str(p_entity.leave_time)])
            df_path = pd.DataFrame(data=p_data, columns=['eid', 'enter_time', 'leave_time'])
            tmp_filename = str(index)+'_'+filename
            df_path.to_csv(os.path.join(target_path_path, tmp_filename), index=False)
