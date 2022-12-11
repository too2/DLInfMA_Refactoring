import os
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
from shapely.geometry import Point, LineString
import geopandas as gpd
import warnings

warnings.filterwarnings('ignore')

def extract_return_loc():
    return_plan = pd.read_csv('/Volumes/T7/TDCT/return_planno.csv')
    return_plan_dict = {}  # key: plan_no, value: location
    for index, value in return_plan.iterrows():
        plan_no = value['plan_no']
        longitude, latitude = value['longitude'], value['latitude']
        return_plan_dict[plan_no] = (longitude, latitude)
    return return_plan_dict


return_plan_dict = extract_return_loc()
np.save('return_plan_dict.npy', return_plan_dict)

# load_dict = np.load('return_plan_dict.npy', allow_pickle=True)
# print(type(load_dict))
