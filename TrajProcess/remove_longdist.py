import pandas as pd
import os
import utils
from objclass import TrajPoint, Traj, TrajAll
from tqdm import tqdm
from time import time
DIST_THRESHOLD = 1000
DIST_ORI_THRESHOLD = 2000
SPEED_THRESHOLD = 33.3

# Extract shandong trajectory
t0 = time()
path = '/Volumes/T7/TDCT'
filename = 'traj_DD.csv'
traj_data = pd.read_csv(os.path.join(path, filename))
t1 = time()
print(t1-t0, 's')


def traj_denoising(traj_data):
    traj_all = TrajAll()
    for plan_no, plan_no_traj_data in tqdm(traj_data.groupby('plan_no')):
        trajectory = Traj(plan_no)
        dri_id = plan_no_traj_data['driver_id'].values[0]
        trajectory.dri_id = dri_id
        point_list = []
        for index, traj in plan_no_traj_data.iterrows():
            point = TrajPoint(traj['plan_no'], traj['longitude'], traj['latitude'], traj['time'])
            point_list.append(point)
        trajectory.traj_point_list = point_list
        traj_all.traj_dict[plan_no] = trajectory

    """
    # Remove the trajectory points within 2km of the origin
    # origin：日照钢厂（119.35426,35.15754）；destination：泰安钢材市场（117.06392,36.07603）
    """
    start_point = TrajPoint(None, 119.35426, 35.15754, None)
    clean_traj_list = []
    for plan_no, traj_obj in tqdm(traj_all.traj_dict.items()):
        start_index = 0
        end_index = len(traj_obj.traj_point_list)
        for index, traj_point_obj in enumerate(traj_obj.traj_point_list):
            curr_point = traj_point_obj
            dist_start = utils.haversine_distance(start_point, curr_point)
            if dist_start > DIST_ORI_THRESHOLD:
                continue
            else:
                start_index = index
                break
        traj_obj.traj_point_list = traj_obj.traj_point_list[start_index:end_index]


    # Trajectory denoising, remove drift trajectory points
    save_traj = []
    for plan_no, traj_obj in tqdm(traj_all.traj_dict.items()):
        last_point = None
        re_index = []
        for index, traj_point_obj in enumerate(traj_obj.traj_point_list):
            if index == 0:
                curr_point = traj_point_obj
                last_point = curr_point
            else:
                curr_point = traj_point_obj
                dist = utils.haversine_distance(curr_point, last_point)
                diff_time = (pd.to_datetime(curr_point.time) - pd.to_datetime(last_point.time)).total_seconds()
                speed = dist / diff_time
                if speed > SPEED_THRESHOLD:
                    re_index.append(index)
                else:
                    last_point = curr_point
        if len(re_index) != 0:
            traj_obj.traj_point_list = [traj_obj.traj_point_list[i] for i in range(len(traj_obj.traj_point_list))
                                        if i not in re_index]

    # Trajectory denoising, remove trajectory with large trajectory point spacing
    save_traj = []
    for plan_no, traj_obj in tqdm(traj_all.traj_dict.items()):
        flag = False
        last_point = None
        if len(traj_obj.traj_point_list) == 0:
            continue
        for index, traj_point_obj in enumerate(traj_obj.traj_point_list):
            if index == 0:
                curr_point = traj_point_obj
                last_point = curr_point
            else:
                curr_point = traj_point_obj
                dist = utils.haversine_distance(curr_point, last_point)
                if dist > DIST_THRESHOLD:
                    flag = True
                    break
                last_point = curr_point
        if not flag:
            save_traj.append(traj_obj)

    # Calculate the distance traveled from the origin to the current point
    last_point = None
    for plan_no, traj_obj in tqdm(traj_all.traj_dict.items()):
        for index, traj_point_obj in enumerate(traj_obj.traj_point_list):
            if index == 0:
                curr_point = traj_point_obj
                last_point = curr_point
                traj_point_obj.dist = 0
            else:
                curr_point = traj_point_obj
                dist = utils.haversine_distance(curr_point, last_point)
                curr_point.dist = last_point.dist + dist
                last_point = curr_point


    save_data = []
    for traj in tqdm(save_traj):
        plan_no = traj.plan_no
        dri_id = traj.dri_id
        traj_list = traj.traj_point_list
        for point in traj_list:
            longitude = point.longitude
            latitude = point.latitude
            timestamp = point.time
            dist = point.dist
            save_data.append([plan_no, dri_id, longitude, latitude, timestamp, dist])

    df_save = pd.DataFrame(data=save_data, columns=['plan_no', 'dri_id', 'longitude',
                                                    'latitude', 'time', 'dist'])
    return df_save

if __name__ == '__main__':
    df_save = traj_denoising(traj_data)
    save_path = '/Volumes/T7/TDCT'
    save_filename = 'new_traj_flow.csv'
    df_save.to_csv(os.path.join(save_path, save_filename), index=False)
