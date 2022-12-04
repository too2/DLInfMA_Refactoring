"""
Read data files, include trajectory data, waybill data, destination library data and road network data
"""
import pandas as pd
import os
import time
import matplotlib.pyplot as plt

def preprocess_way2endname():
    # read destination library data
    file_path = '/Volumes/T7/traj_file/'
    """
    Read waybill&destination data
    data format: 
    plan_no	        waybill_no	    end_point	city_name
    DD201121000778	YD201121001661	P000016646	泰安市
    """
    way2endname_1112 = pd.read_csv(os.path.join(file_path, 'way2endname_1112.csv'))
    way2endname_0102 = pd.read_csv(os.path.join(file_path, 'way2endname_0102.csv'))
    way2endname_0305 = pd.read_csv(os.path.join(file_path, 'way2endname_0305.csv'))
    way2endname_all = pd.concat([way2endname_1112, way2endname_0102, way2endname_0305], axis=0)

    # Extract the plan_no , including 'DD' and the destination in Shandong Province
    shandong_city = ['济南', '青岛', '淄博', '枣庄', '东营', '烟台', '潍坊', '济宁', '泰安', '威海', '日照', '临沂',
                     '德州', '聊城', '滨州', '菏泽']
    way2endname_DD = way2endname_all[(way2endname_all['plan_no'].str[:2] == 'DD') &
                                     (way2endname_all['city_name'].str[:2].isin(shandong_city))]

    # Delete the plan_no containing multiple waybills  (94825 --> 86508)
    way2endname_DD_unique = way2endname_DD.drop_duplicates(subset=['plan_no'], keep=False)
    return way2endname_DD_unique

def extract_traj():
    # Extract selected trajectory
    file_path = '/Volumes/T7/traj_file/'
    way2endname_DD_unique = preprocess_way2endname()
    t1 = time.time()
    se_plan_no = set(way2endname_DD_unique['plan_no'])

    traj_1_2 = pd.read_csv(os.path.join(file_path, 'trajctory1-2.csv'), usecols=['plan_no', 'truck_no', 'time',
                                                                                 'longitude', 'latitude', 'speed'])
    traj_11_12 = pd.read_csv(os.path.join(file_path, 'trajctory11-12.csv'), usecols=['plan_no', 'truck_no', 'time',
                                                                                     'longitude', 'latitude', 'speed'])
    traj_3_5 = pd.read_csv(os.path.join(file_path, 'trajctory3-5.csv'), usecols=['plan_no', 'truck_no', 'time',
                                                                                 'longitude', 'latitude', 'speed'])
    traj_0102 = traj_1_2[traj_1_2['plan_no'].isin(se_plan_no)]
    del traj_1_2
    traj_0305 = traj_3_5[traj_3_5['plan_no'].isin(se_plan_no)]
    del traj_3_5
    traj_1112 = traj_11_12[traj_11_12['plan_no'].isin(se_plan_no)]
    del traj_11_12
    t2 = time.time()
    print(t2-t1, 's')

    # Read waybill data, merge trajectory data and add driver_id field
    waybill_data = pd.read_csv(os.path.join(file_path, 'waybill_data_name.csv'))
    traj_flow = pd.concat([traj_1112, traj_0102, traj_0305], axis=0)
    traj_flow = pd.merge(traj_flow, waybill_data[['plan_no', 'driver_id']], on='plan_no', how='left')
    traj_flow.info()

    # Delete trajectories with too many or too few trajectory points
    tmp = traj_flow['plan_no'].value_counts().tolist()
    plt.hist(tmp, bins=40, facecolor="blue", edgecolor="black", alpha=0.7, range=(100, 4000)
             )  # weights=np.ones(len(tmp)) / len(tmp)
    traj_flow_re_longshort = traj_flow.groupby('plan_no').filter(lambda x: 100 <= len(x) <= 4000)
    return traj_flow_re_longshort

if __name__ == '__main__':
    """
    Save trajectory
    data format: 
    plan_no         truck_no  time                 longitude   latitude   speed   waybill_no      driver_id  
    DD201121000778  鲁Q3M835  2020-11-21 10:32:02  119.114335  34.60401    0.0    YD201121001661  U000001028    
    """
    traj_flow_re_longshort = extract_traj()
    traj_flow_re_longshort.to_csv('/Volumes/T7/TDCT/traj_DD.csv', index=False)

