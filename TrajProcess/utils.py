from math import radians, cos, sin, asin, sqrt, atan, pi, log, tan, exp, atan2, degrees, fabs
import numpy as np
from scipy import interpolate
from pyproj import Transformer
import math

DEGREES_TO_RADIANS = math.pi / 180
RADIANS_TO_DEGREES = 1 / DEGREES_TO_RADIANS
EARTH_MEAN_RADIUS_METER = 6371008.7714
DEG_TO_KM = DEGREES_TO_RADIANS * EARTH_MEAN_RADIUS_METER
LAT_PER_METER = 8.993203677616966e-06
LNG_PER_METER = 1.1700193970443768e-05


def haversine_distance(a, b):
    if same_coords(a, b):
        return 0.0
    delta_lat = math.radians(b.latitude - a.latitude)
    delta_lng = math.radians(b.longitude - a.longitude)
    h = math.sin(delta_lat / 2.0) * math.sin(delta_lat / 2.0) + math.cos(math.radians(a.latitude)) * math.cos(
        math.radians(b.latitude)) * math.sin(delta_lng / 2.0) * math.sin(delta_lng / 2.0)
    c = 2.0 * math.atan2(math.sqrt(h), math.sqrt(1 - h))
    d = EARTH_MEAN_RADIUS_METER * c
    return d


def same_coords(a, b):
    if a.latitude == b.latitude and a.longitude == b.longitude:
        return True
    else:
        return False


def wgs84_to_mercator(lon, lat):
    x = lon * 20037508.342789 / 180
    y = log(tan((90 + lat) * pi / 360)) / (pi / 180)
    y = y * 20037508.34789 / 180
    return [x, y]


def mercator_to_wgs84(x, y):
    lon = x / 20037508.34 * 180
    lat = y / 20037508.34 * 180
    lat = 180 / pi * (2 * atan(exp(lat * pi / 180)) - pi / 2)
    return [lon, lat]


def haversine_distance_loc_points(lng1, lat1, lng2, lat2):
    delta_lat = math.radians(lat2 - lat1)
    delta_lng = math.radians(lng2 - lng1)
    h = math.sin(delta_lat / 2.0) * math.sin(delta_lat / 2.0) + math.cos(math.radians(lat1)) * math.cos(
        math.radians(lat2)) * math.sin(delta_lng / 2.0) * math.sin(delta_lng / 2.0)
    c = 2.0 * math.atan2(math.sqrt(h), math.sqrt(1 - h))
    d = EARTH_MEAN_RADIUS_METER * c
    return d


def utm_to_wgs84(lng, lat):
    # WGS84 = Proj(init='EPSG:4326')
    # p = Proj(init="EPSG:32650")
    # x,y = lng, lat
    # return transform(p, WGS84, x, y)
    transformer = Transformer.from_crs("epsg:32650", "epsg:4326")
    a = transformer.transform(lng, lat)
    print(a)
    return [a[1], a[0]]


# coordinate tool
def _transform_lat(lng, lat):
    ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + 0.1 * lng * lat + 0.2 * sqrt(fabs(lng))
    ret += (20.0 * sin(6.0 * lng * pi) + 20.0 * sin(2.0 * lng * pi)) * 2.0 / 3.0
    ret += (20.0 * sin(lat * pi) + 40.0 * sin(lat / 3.0 * pi)) * 2.0 / 3.0
    ret += (160.0 * sin(lat / 12.0 * pi) + 320 * sin(lat * pi / 30.0)) * 2.0 / 3.0
    return ret


def _transform_lng(lng, lat):
    ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + 0.1 * lng * lat + 0.1 * sqrt(fabs(lng))
    ret += (20.0 * sin(6.0 * lng * pi) + 20.0 * sin(2.0 * lng * pi)) * 2.0 / 3.0
    ret += (20.0 * sin(lng * pi) + 40.0 * sin(lng / 3.0 * pi)) * 2.0 / 3.0
    ret += (150.0 * sin(lng / 12.0 * pi) + 300.0 * sin(lng / 30.0 * pi)) * 2.0 / 3.0
    return ret


def out_of_china(lng, lat):
    """
    判断是否在国内，不在国内不做偏移
    :param lng:
    :param lat:
    :return:
    """
    return not (73.66 < lng < 135.05 and 3.86 < lat < 53.55)


def wgs84_to_gcj02(lng, lat):
    """
    WGS84转GCJ02(火星坐标系)
    :param lng:WGS84坐标系的经度
    :param lat:WGS84坐标系的纬度
    :return:
    """
    if out_of_china(lng, lat):  # 判断是否在国内
        return [lng, lat]
    dlat = _transform_lat(lng - 105.0, lat - 35.0)
    dlng = _transform_lng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * pi
    magic = sin(radlat)
    magic = 1 - 0.00669342162296594323 * magic * magic
    sqrtmagic = sqrt(magic)
    dlat = (dlat * 180.0) / ((6378245.0 * (1 - 0.00669342162296594323)) / (magic * sqrtmagic) * pi)
    dlng = (dlng * 180.0) / (6378245.0 / sqrtmagic * cos(radlat) * pi)
    return lat + dlat, lng + dlng


def gcj02_to_wgs84(lng, lat):
    """
    GCJ02(火星坐标系)转GPS84
    :param lng:火星坐标系的经度
    :param lat:火星坐标系纬度
    :return:
    """
    if out_of_china(lng, lat):
        return [lng, lat]
    dlat = _transform_lat(lng - 105.0, lat - 35.0)
    dlng = _transform_lng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * pi
    magic = 1 - 0.00669342162296594323 * sin(radlat) * sin(radlat)
    sqrtmagic = sqrt(magic)
    dlat = (dlat * 180.0) / ((6378245.0 * (1 - 0.00669342162296594323)) / (magic * sqrtmagic) * pi)
    dlng = (dlng * 180.0) / (6378245.0 / sqrtmagic * cos(radlat) * pi)
    mglat = lat + dlat
    mglng = lng + dlng
    return lng * 2 - mglng, lat * 2 - mglat


# def wgs84_to_mercator(lon, lat):
#     x = lon * 20037508.342789 / 180
#     y = log(tan((90 + lat) * pi / 360)) / (pi / 180)
#     y = y * 20037508.34789 / 180
#     return [x, y]
#
#
# def mercator_to_wgs84(x, y):
#     lon = x / 20037508.34 * 180
#     lat = y / 20037508.34 * 180
#     lat = 180 / pi * (2 * atan(exp(lat * pi / 180)) - pi / 2)
#     return [lon, lat]


def transform_points_wgs84_to_mercator(coordinates):
    temp_result = []
    for i, item in enumerate(coordinates):
        temp_result.append(wgs84_to_mercator(item[0], item[1]))
    return temp_result


def transform_points_mercator_to_wgs84(coordinates):
    temp_result = []
    for i, item in enumerate(coordinates):
        temp_result.append(mercator_to_wgs84(item[0], item[1]))
    return temp_result


def transform_points_wgs84_to_gcj02(coordinates):
    temp_result = []
    for i, item in enumerate(coordinates):
        temp_result.append(wgs84_to_gcj02(item[0], item[1]))
    return temp_result


def transform_points_gcj02_to_wgs84(coordinates):
    temp_result = []
    for i, item in enumerate(coordinates):
        temp_result.append(gcj02_to_wgs84(item[0], item[1]))
    return temp_result


# angel tool    参考三面角余弦定理，书签博客
def get_angle_to_north(lon1, lat1, lon2, lat2):
    lat1_radians = radians(lat1)  # radians 角度转弧度
    lon1_radians = radians(lon1)
    lat2_radians = radians(lat2)
    lon2_radians = radians(lon2)
    lon_difference = lon2_radians - lon1_radians
    y = sin(lon_difference) * cos(lat2_radians)
    x = cos(lat1_radians) * sin(lat2_radians) - sin(lat1_radians) * cos(lat2_radians) * cos(lon_difference)
    return (degrees(atan2(y, x)) + 360) % 360  # atan2 反正切   degrees 弧度转角度


def calculate_angle_diff(angel_diff):
    abs_angel_diff = fabs(angel_diff)
    if abs_angel_diff > 180:
        return 360 - abs_angel_diff
    else:
        return abs_angel_diff


def get_cos_value(net, link):
    vector_a = np.mat(net)
    vector_b = np.mat(link)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    return num / denom


def detect_u_turn(shape, u_turn_angle=130):
    start_angle = get_angle_to_north(shape[0][0], shape[0][1], shape[1][0], shape[1][1])
    end_angle = get_angle_to_north(shape[-2][0], shape[-2][1], shape[-1][0], shape[-1][1])
    return calculate_angle_diff(start_angle - end_angle) > u_turn_angle


def projection_direction(direction):
    para_length = 100.0
    if 0 <= direction < 90:
        return [sin(radians(direction)) * para_length,
                cos(radians(direction)) * para_length]
    if 90 <= direction < 180:
        return [sin(radians(direction)) * para_length,
                -cos(radians(direction)) * para_length]
    if 180 <= direction < 270:
        return [-sin(radians(direction)) * para_length,
                -cos(radians(direction)) * para_length]
    if 270 <= direction < 360:
        return [-sin(radians(direction)) * para_length,
                cos(radians(direction)) * para_length]


# distance tool
def eucl_distance(x, y):
    return np.linalg.norm(x - y)




