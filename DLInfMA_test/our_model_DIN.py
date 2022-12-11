import pandas as pd
from torch import nn
import torch
from torch.utils.data import DataLoader, Dataset
from extract_feature import extract_feature
import numpy as np
import pickle
import math
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.model_selection import train_test_split
from metrics import metrics
from collections import defaultdict
from tqdm import tqdm
import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
# 设置随机数种子
setup_seed(20)

# # 提取特征和标签
# address_dict = extract_feature()


with open('address_dict.pickle', 'rb') as f:
    address_dict = pickle.load(f)


def get_combin_candi(addressobj, train=True):
    if train:
        combin_candi = []
        if len(addressobj.candiates_list) == 1:
            candi = addressobj.candiates_list[0]
            return [[candi, candi, addressobj, 1]]
        for candi1 in addressobj.candiates_list:
            if candi1.sr_id != addressobj.candiates_label_index:
                continue
            else:
                for candi2 in addressobj.candiates_list:
                    if candi1.sr_id == candi2.sr_id:
                        continue
                    else:
                        combin_candi.append([candi1, candi2, addressobj, 1])
                        combin_candi.append([candi2, candi1, addressobj, 0])
    else:
        combin_candi = []
        if len(addressobj.candiates_list) == 1:
            candi = addressobj.candiates_list[0]
            return [[candi, candi, addressobj]]
        for candi1 in addressobj.candiates_list[:-1]:
            for candi2 in addressobj.candiates_list[1:]:
                if candi1.sr_id == candi2.sr_id:
                    continue
                else:
                    combin_candi.append([candi1, candi2, addressobj])
    return combin_candi


# 候选位置组合
train_ratio = 0.7
train_data_id, test_data_id = train_test_split([i for i in range(len(address_dict))], test_size=0.3)
train_combin_candi = []
test_combin_candi = []
for index, value in tqdm(enumerate(address_dict.items())):
    end_point, addressobj = value[0], value[1]
    if index in train_data_id:
        train_combin_candi.extend(get_combin_candi(addressobj, True))
    else:
        test_combin_candi.extend(get_combin_candi(addressobj, False))


def get_candi_feature(candi, end_point):
    feature = []
    # behavior feature
    behav_cargo_type = [len(candi.cargo_type_set)]  # 先不考虑类别，只考虑种类数量（长度为1）
    behav_start_time = candi.time_distribution  # 考虑时间分布（长度为24）
    behav_time_dura = [candi.avg_duration]  # 考虑停留时长（长度为1）
    behav_stop_fre = candi.stop_fre  # 考虑停留频次（长度为4）
    behav_trip_coverage = [candi.trip_converage[end_point]]  # 考虑行程覆盖率（长度为1）
    # area feature
    area_dist = [candi.dist_end[end_point]]  # 考虑距离终点距离 （长度为1）
    candi.nearPoiType_distribute = candi.poi_type
    area_road_type = [0]  # 考虑邻近道路类型 candi.nearRoad_type
    area_poi_distr = candi.nearPoiType_distribute  # 考虑邻近POI类别分布（长度为3, 全连接）
    area_area = [candi.area]  # 考虑候选位置形成凸包面积（长度为1）

    [feature.extend(i) for i in [behav_cargo_type, behav_start_time, behav_time_dura, behav_stop_fre,
                                 behav_trip_coverage, area_dist, area_road_type, area_poi_distr,
                                 area_area]]
    return feature


def get_address_feature(addressobj):
    feature = []
    # address feature
    addressobj.nearPoiType_distribute = addressobj.poi_type
    address_poi_distr = addressobj.nearPoiType_distribute  # 考虑地址邻近POI类型分布（反编码对应位置，长度为3）
    address_trip_num = [len(addressobj.trip_set)]  # 考虑地址行程数量
    [feature.extend(i) for i in [address_poi_distr, address_trip_num]]
    return feature


def get_sample(is_train='train'):
    feature = []
    tag = []
    if is_train == 'train':
        combin_candi = train_combin_candi
        for index, f in tqdm(enumerate(combin_candi)):
            candi1, candi2, addressobj, relation = f[0], f[1], f[2], f[3]
            end_point = addressobj.end_point
            tmp_feature = get_candi_feature(candi1, end_point)
            tmp_feature.extend(get_candi_feature(candi2, end_point))
            tmp_feature.extend(get_address_feature(addressobj))
            feature.append(tmp_feature)
            tag.append(relation)
    else:
        combin_candi = test_combin_candi
        for index, f in tqdm(enumerate(combin_candi)):
            candi1, candi2, addressobj = f[0], f[1], f[2]  # 包含候选位置1、候选位置2、终点ID
            end_point = addressobj.end_point
            tmp_feature = get_candi_feature(candi1, end_point)
            tmp_feature.extend(get_candi_feature(candi2, end_point))
            tmp_feature.extend(get_address_feature(addressobj))
            feature.append(tmp_feature)
            tag.append([candi1.sr_id, candi2.sr_id, data_end_point2num[addressobj.end_point]])
    return feature, tag


# end_point--num映射
data_end_point2num = {}
# num--addressobj映射
data_num2end_point = {}
for index, value in enumerate(address_dict.items()):
    end_point = value[0]
    addressobj = value[1]
    data_end_point2num[addressobj.end_point] = index
    data_num2end_point[index] = addressobj

feature_train, tag_train = get_sample(is_train='train')
feature_test, tag_test = get_sample(is_train='test')


class MyDataset(Dataset):
    def __init__(self, is_train=True):
        super().__init__()
        if is_train:
            self.feature = feature_train
            self.tag = tag_train
        else:
            self.feature = feature_test
            self.tag = tag_test

    def __getitem__(self, index):
        return self.feature[index], self.tag[index]

    def __len__(self):
        return len(self.feature)


batch_size = 16
data_train = MyDataset(is_train=True)
data_test = MyDataset(is_train=False)
data_loader_train = DataLoader(data_train, batch_size=batch_size, shuffle=True)
data_loader_test = DataLoader(data_test, batch_size=batch_size, shuffle=True)


class FeaAttCal(nn.Module):
    def __init__(self, num_behav, num_area, num_address_feature, emb_hidden_size):
        super().__init__()
        # self.linear_behav = nn.Linear(num_behav, emb_hidden_size)
        # self.linear_area = nn.Linear(num_area, emb_hidden_size)
        # self.linear_address = nn.Linear(num_address_feature, emb_hidden_size)
        self.model = nn.Sequential(
            nn.Linear(41, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        data = torch.stack(data).numpy().T
        data = data.astype(np.float32) / (data.max(axis=0).astype(np.float32)+1e-8)
        behav_feature1 = torch.from_numpy(data[:, :31]).to(torch.float32)
        area_feature1 = torch.from_numpy(data[:, 31:37]).to(torch.float32)
        behav_feature2 = torch.from_numpy(data[:, 37:68]).to(torch.float32)
        area_feature2 = torch.from_numpy(data[:, 68:74]).to(torch.float32)
        address_feature = torch.from_numpy(data[:, 74:]).to(torch.float32)

        # behav_feature1 = self.relu(self.bn1(self.linear_behav(behav_feature1)))
        # area_feature1 = self.relu(self.bn2(self.linear_area(area_feature1)))
        # behav_feature2 = self.relu(self.bn1(self.linear_behav(behav_feature2)))
        # area_feature2 = self.relu(self.bn2(self.linear_area(area_feature2)))
        # address_feature = self.linear_address(address_feature)

        # feature_set1 = [i.unsqueeze(1) for i in [behav_feature1, area_feature1, address_feature]]
        # feature_set2 = [i.unsqueeze(1) for i in [behav_feature2, area_feature2, address_feature]]
        # feature_set1 = torch.cat(feature_set1, dim=1)
        # feature_set2 = torch.cat(feature_set2, dim=1)

        feature_set1 = torch.cat((behav_feature1, area_feature1, address_feature), dim=1)
        feature_set2 = torch.cat((behav_feature2, area_feature2, address_feature), dim=1)
        # feature_set, weight = self.attention(address_trip_num, feature_set, feature_set)
        # x1 = torch.flatten(feature_set1, start_dim=1)
        x1 = feature_set1
        x1 = self.model(x1)
        x2 = feature_set2
        x2 = self.model(x2)
        diff = x1 - x2
        prob = self.sigmoid(diff)
        return prob

    def attention(self, q, k, v):
        q = q.unsqueeze(1)
        weight = torch.matmul(k, q.permute(0, 2, 1))
        weight = self.sigmoid(weight)
        return (1+weight) * v, weight


model = FeaAttCal(num_behav=31, num_area=6, num_address_feature=4, emb_hidden_size=8)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
print(model)

# optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()


def train(epoch):
    model.train()
    train_loss = 0
    correct = 0
    sample_num = 0
    for feature, tag in data_loader_train:
        if torch.cuda.is_available():
            feature = feature.cuda()
            tag = tag.cuda()
        optimizer.zero_grad()
        output = model(feature).squeeze()
        loss = criterion(output, tag.float())
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(feature[0])
        prediction = torch.tensor([1 if i >= 0.5 else 0 for i in output])
        correct += torch.sum(prediction == tag)
        sample_num += len(prediction)
    train_loss = train_loss / len(data_loader_train.dataset)
    acc = correct / sample_num
    # print('Epoch: {} \tTraining Loss: {}'.format(epoch, train_loss))
    print('Epoch: {} \tTraining Loss: {:.6f}, Acc:: {}'.format(epoch, train_loss, acc))
    return train_loss

def val(epoch):
    model.eval()
    ans_output = []
    ans_pair = []
    for feature, pair in data_loader_test:
        if torch.cuda.is_available():
            feature = feature.cuda()
        output = model(feature)
        ans_output.extend(output)
        pair = torch.stack(pair).permute(1, 0)
        ans_pair.extend(pair)
    return ans_output, ans_pair


def get_real_return_id(tag_return_candi_id, ans):
    vote_candi = defaultdict(int)
    for p, a in zip(tag_return_candi_id, ans):
        p1, p2 = p[0], p[1]
        if a == 1:
            vote_candi[p1] += 1
        else:
            vote_candi[p2] += 1
    # 找到投票分数最高的id值
    pred_best_id = None
    for key, value in vote_candi.items():
        if value == max(vote_candi.values()):
            pred_best_id = key
            break
    return pred_best_id

loss_train = []
loss_test = []
epochs = 100
best_epoch = None
for epoch in range(epochs):
    loss = train(epoch)
    loss_train.append(loss)
    ans_output, ans_pair = val(epoch)
    # 构造数据结构
    # candi1--candi2--end_point--output
    ans_output = torch.tensor([1 if i >= 0.5 else 0 for i in ans_output])
    ans_pair = torch.stack(ans_pair)
    ans = torch.cat((ans_pair, ans_output.reshape(-1, 1)), dim=-1).numpy()
    df = pd.DataFrame(data=ans, columns=['candi1', 'candi2', 'end_point', 'output'])
    label_loc_list = []
    pred_loc_list = []
    for end_point, value in df.groupby('end_point'):
        tag_return_candi_id = value[['candi1', 'candi2']].values.tolist()
        output = value['output'].values.tolist()
        pred_best_id = get_real_return_id(tag_return_candi_id, output)
        addressobj = data_num2end_point[end_point]
        label_loc = addressobj.candiates_list[addressobj.candiates_label_index]
        pred_loc = None
        for candi in addressobj.candiates_list:
            if candi.sr_id == pred_best_id:
                pred_loc = candi
                break
        label_loc_list.append([label_loc.cen_lng, label_loc.cen_lat])
        pred_loc_list.append([pred_loc.cen_lng, pred_loc.cen_lat])

    metric = metrics(label_loc_list, pred_loc_list)
    print(metric)


