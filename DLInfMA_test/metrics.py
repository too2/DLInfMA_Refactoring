# 几种评价指标
# MAE、P85、beta500、beta1000、beta3000
from utils_poi import geo_distance
import numpy as np
import math

def metrics(label_loc_list, pred_loc_list):
    dist_list = []
    for label, pred in zip(label_loc_list, pred_loc_list):
        P1_lat, P1_lng, P2_lat, P2_lng = label[1], label[0], pred[1], pred[0]
        dist = geo_distance(P1_lat, P1_lng, P2_lat, P2_lng)
        dist_list.append(dist)
    mae = np.mean(dist_list)
    p85 = sorted(dist_list)[math.floor(len(dist_list)*0.85)]
    beta500 = list((np.array(dist_list) <= 500)).count(True) / len(dist_list)
    beta1000 = list((np.array(dist_list) <= 1000)).count(True) / len(dist_list)
    beta3000 = list((np.array(dist_list) <= 3000)).count(True) / len(dist_list)
    return mae, p85, beta500, beta1000, beta3000
