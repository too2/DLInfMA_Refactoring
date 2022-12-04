from common.road_network import load_rn_shp
from common.trajectory import Trajectory, store_traj_file, parse_traj_file
from common.trajectory import STPoint
from map_matching.hmm.hmm_map_matcher import TIHMMMapMatcher
from datetime import datetime
import os
from tqdm import tqdm
import argparse
from multiprocessing import Pool
import time

os.environ['PROJ_LIB'] = '/home/mjl/anaconda3/envs/ox/lib/python3.9/site-packages/pyproj/proj_dir/share/proj'
os.environ['GDAL_DATA'] = '/home/mjl/anaconda3/envs/ox/lib/python3.9/site-packages/pyproj/proj_dir/share'

def parse_tdrive(filename, tdrive_root_dir):
    oid = filename.replace('.txt', '')
    # oid = filename.replace('.csv', '')
    with open(os.path.join(tdrive_root_dir, filename), 'r') as f:
        pt_list = []
        for index, line in enumerate(f.readlines()):
            if index == 0:
                continue
            attrs = line.strip('\n').split(',')
            lat = float(attrs[3])
            lng = float(attrs[2])
            time = datetime.strptime(attrs[1], '%Y-%m-%d %H:%M:%S')
            pt_list.append(STPoint(lat, lng, time))
    if len(pt_list) > 1:
        return Trajectory(oid, 0, pt_list)
    else:
        return None


def mm_tdrive(clean_traj_dir, mm_traj_dir, mm_traj_path_dir, rn_path):
    param_list = []
    for index, filename in tqdm(enumerate(os.listdir(clean_traj_dir))):
        if filename[0] == '.':
            continue
        clean_trajs = parse_traj_file(os.path.join(clean_traj_dir, filename))
        param_list.append([clean_trajs, rn_path, mm_traj_dir, mm_traj_path_dir, filename])

    print("开始匹配")
    t1 = time.time()
    with Pool(90) as pool:
        progress_bar = tqdm(total=len(param_list))
        list(tqdm(pool.imap(do_mm, param_list), total=len(param_list)))
    t2 = time.time()
    print("并行执行时间：{}s".format(int(t2-t1)))


def do_mm(param):
    clean_trajs = param[0]
    rn_path = param[1]
    mm_traj_dir = param[2]
    mm_traj_path_dir = param[3]
    filename = param[4]
    rn = load_rn_shp(rn_path, is_directed=True)
    map_matcher = TIHMMMapMatcher(rn)
    mm_trajs = [map_matcher.match_to_path(clean_traj) for clean_traj in clean_trajs]
    store_traj_file(mm_trajs, os.path.join(mm_traj_dir, filename), mm_traj_path_dir, filename, traj_type='mm')


def mm_tdrive_test(clean_traj_dir, mm_traj_dir, rn_path, mm_traj_path_dir):
    rn = load_rn_shp(rn_path, is_directed=True)
    map_matcher = TIHMMMapMatcher(rn)
    for filename in tqdm(os.listdir(clean_traj_dir)):
        clean_trajs = parse_traj_file(os.path.join(clean_traj_dir, filename))
        mm_trajs = [map_matcher.match_to_path(clean_traj) for clean_traj in clean_trajs]
        store_traj_file(mm_trajs, os.path.join(mm_traj_dir, filename), mm_traj_path_dir, filename,
                        traj_type='mm')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_traj_dir', help='the directory of the cleaned trajectories')
    parser.add_argument('--rn_path', help='the road network data path generated by osm2rn')
    parser.add_argument('--mm_traj_dir', help='the directory of the map-matched trajectories')
    parser.add_argument('--mm_traj_path_dir', help='the directory of the map-matched paths')
    parser.add_argument('--phase', help='the preprocessing phase [clean,mm,stat]')

    opt = parser.parse_args()
    opt.phase = 'mm'
    print(opt)


    # 进行数据清洗
    if opt.phase == 'mm':
        # 文件目录格式
        # cluster_id
        # - rn_path
        # - traj_cleaning
        # - save_mm_file
        path = '../data' #'/Volumes/T7/TDCT/'
        for id in range(len(os.listdir(os.path.join(path, 'osm_file')))):
            opt.clean_traj_dir = os.path.join(path, 'traj_cleaning/'+str(id)+'_cluster_id')
            opt.mm_traj_dir = os.path.join(path, 'save_mm_traj_point/'+str(id)+'_cluster_id')
            opt.mm_traj_path_dir = os.path.join(path, 'save_mm_traj_path/'+str(id)+'_cluster_id')
            if not os.path.exists(opt.mm_traj_dir):
                os.makedirs(opt.mm_traj_dir)
            if not os.path.exists(opt.mm_traj_path_dir):
                os.makedirs(opt.mm_traj_path_dir)
            opt.rn_path = os.path.join(path, 'osm_file/'+str(id)+'_cluster_id'+'/edges.shp')
            mm_tdrive(opt.clean_traj_dir, opt.mm_traj_dir, opt.mm_traj_path_dir, opt.rn_path)
            print('--------------------------------------------------', id)

        # # 不用多进程进行测试
        # opt.rn_path = '/Volumes/T7/TDCT/osm_file/0_cluster_id/edges.shp'
        # opt.clean_traj_dir = '/Volumes/T7/TDCT/traj_cleaning_test/0_cluster_id/'
        # opt.mm_traj_dir = '/Volumes/T7/TDCT/save_mm_traj_point'
        # mm_traj_path_dir = '/Volumes/T7/TDCT/save_mm_traj_path'
        # filename = 'DD210330002337.csv'
        # mm_tdrive_test(opt.clean_traj_dir, opt.mm_traj_dir, opt.rn_path, mm_traj_path_dir)
    else:
        raise Exception('unknown phase')