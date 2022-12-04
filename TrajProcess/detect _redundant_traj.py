"""
Detection of redundant traj，which means check whether the timestamp of the last point of the traj
exceeds the return time
"""
import pandas as pd
from tqdm import tqdm
from time import time

# Change the return time of traj record with 'alter_time' field
t0 = time()
data = pd.read_csv('/Volumes/T7/traj_file/trajctory3-5.csv')
data = data[data['plan_no'].str[:2] == 'DD']
data = data.drop_duplicates(subset='plan_no', keep='last')
data[['create_time', 'load_date', 'return_time', 'time']] = data[['create_time', 'load_date', 'return_time',
                                                                  'time']].apply(lambda x: pd.to_datetime(x))
t1 = time()
print(t1-t0, 's')
print(list(data.columns))

data = data.reset_index(drop=True)
ans = 0
for plan_no, value in tqdm(data.groupby('plan_no')):
    for index, v in value.iterrows():
        if isinstance(v['alter_time'], str):  # and not math.isnan(v['alter_time'])
            v['alter_time'] = pd.to_datetime(v['alter_time'])
            v['return_time'] = v['alter_time']
        if v['time'] > v['return_time']:
            print(index, plan_no)
            ans += 1
            break
print(ans)
