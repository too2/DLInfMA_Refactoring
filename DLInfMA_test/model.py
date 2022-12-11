from torch import nn
import torch
from torch.utils.data import DataLoader, Dataset
from extract_feature import extract_feature
import numpy as np
import pickle
import math
from torch.utils.tensorboard import SummaryWriter
from metrics import metrics
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
#
# with open('address_dict.pickle', 'wb') as f:
#     pickle.dump(address_dict, f)
with open('address_dict.pickle', 'rb') as f:
    address_dict = pickle.load(f)

data_x = []  # 特征
data_loc = []  # 位置坐标
data_y = []  # 标签
data_end_point = []  # 终点ID
end_point_lab_loc = {}  # 记录终点ID地址库位置
end_point_real_loc = {}  # 记录终点ID真实位置
for end_point, addressobj in address_dict.items():  # num_dri 司机数等会进行考虑
    len_candi = len(addressobj.candiates_list)
    trip_coverage = []
    distance = []
    avg_duration = []
    time_distribution = []
    for candiobj in addressobj.candiates_list:
        trip_coverage.append(candiobj.trip_converage[end_point])
        distance.append(candiobj.dist_end[end_point])
        avg_duration.append(candiobj.avg_duration)
        time_distribution.append(tuple(candiobj.time_distribution))
    num_deliveries = [len(addressobj.trip_set)]*len_candi  # 只有这个是针对地址来说的，是一个标量
    lab_lng = addressobj.lab_lng
    lab_lat = addressobj.lab_lat
    end_point_lab_loc[end_point] = (lab_lng, lab_lat)
    sample = []
    for f1, f2, f3, f4, f5 in zip(trip_coverage, distance, avg_duration, num_deliveries, time_distribution):
        tmp = [f1, f2, f3, f4]
        tmp.extend(f5)
        sample.append(tmp)
    tag = addressobj.candiates_label_index
    data_x.append(sample)
    data_y.append(tag)
    data_loc.append([(candi.cen_lng, candi.cen_lat) for candi in addressobj.candiates_list])
    data_end_point.append(end_point)
    end_point_real_loc[end_point] = (addressobj.real_lng, addressobj.real_lat)

with open('data_x.pickle', 'wb') as f:
    pickle.dump(data_x, f)
with open('data_y.pickle', 'wb') as f:
    pickle.dump(data_y, f)
with open('data_loc.pickle', 'wb') as f:
    pickle.dump(data_loc, f)
with open('data_end_point.pickle', 'wb') as f:
    pickle.dump(data_end_point, f)
with open('end_point_lab_loc.pickle', 'wb') as f:
    pickle.dump(end_point_lab_loc, f)
with open('end_point_real_loc.pickle', 'wb') as f:
    pickle.dump(end_point_real_loc, f)

with open('data_x.pickle', 'rb') as f:
    data_x = pickle.load(f)
with open('data_y.pickle', 'rb') as f:
    data_y = pickle.load(f)
with open('data_loc.pickle', 'rb') as f:
    data_loc = pickle.load(f)
with open('data_end_point.pickle', 'rb') as f:
    data_end_point = pickle.load(f)
with open('end_point_lab_loc.pickle', 'rb') as f:
    end_point_lab_loc = pickle.load(f)
with open('end_point_real_loc.pickle', 'rb') as f:
    end_point_real_loc = pickle.load(f)
# print(data_x)
# print(data_y)
print('------')

# end_point--num映射
data_end_point2num = {}
# num--end_point映射
data_num2end_point = {}
for i, j in zip(range(len(data_end_point)), data_end_point):
    data_end_point2num[j] = i
    data_num2end_point[i] = j

data_end_num = [i for i in range(len(data_end_point))]


data_x = np.array(data_x)
print(data_x.shape)

# 采用min-max进行归一化
min_max_data_x = []
for i in data_x:
    for j in i:
        min_max_data_x.append(j)
min_max_data_x = np.array(min_max_data_x)
index_min_data_x = np.argmin(min_max_data_x, axis=0)
index_max_data_x = np.argmax(min_max_data_x, axis=0)
min_data_x = min_max_data_x[index_min_data_x, range(min_max_data_x.shape[1])]
max_data_x = min_max_data_x[index_max_data_x, range(min_max_data_x.shape[1])]

data_x_len = []
for i in data_x:
    data_x_len.append(len(i))
print(max(data_x_len))
print(np.mean(data_x_len))


# 设置句子最长长度
max_len = max(data_x_len)  # math.ceil(50)  # np.mean(data_x_len)
# max_len = 30

# mask掩码构造
mask_pad = np.ones((len(data_x), max_len))
for index, value in enumerate(data_x_len):
    if value < max_len:
        mask_pad[index, :value] = 0
    else:
        mask_pad[index, :] = 0


# padding 填充
data_x_pad = np.zeros((len(data_x), max_len, 28))
for index1, x in enumerate(data_x):
    for index2, value in enumerate(x):
        if index2 >= max_len:
            break
        data_x_pad[index1, index2, :] = (np.array(value)-min_data_x)/max_data_x

# 标签表示
data_y_pad = np.zeros((len(data_y), max_len))
for index, y in enumerate(data_y):
    if y >= max_len:
        continue
    data_y_pad[index, y] = 1

# 标签对应经纬度
data_loc_pad = np.zeros((len(data_loc), max_len, 2))
for index1, loc in enumerate(data_loc):
    for index2, value in enumerate(loc):
        if index2 >= max_len:
            break
        data_loc_pad[index1, index2, :] = np.array(value)


train_ratio = 0.7
x_train, y_train = data_x_pad[:int(len(data_x_pad) * train_ratio)], data_y_pad[:int(len(data_y_pad) * train_ratio)]
x_val, y_val = data_x_pad[int(len(data_x_pad) * train_ratio):], data_y_pad[int(len(data_y_pad) * train_ratio):]
mask_train = mask_pad[:int(len(mask_pad) * train_ratio)]
mask_val = mask_pad[int(len(mask_pad) * train_ratio):]
loc_train, loc_test = data_loc_pad[:int(len(data_loc_pad) * train_ratio)], data_loc_pad[int(len(data_loc_pad) * train_ratio):]
end_num_train, end_num_val = data_end_num[:int(len(data_end_num) * train_ratio)], data_end_num[int(len(data_end_num) * train_ratio):]

class MyDataset(Dataset):
    def __init__(self, is_train=True):
        super().__init__()
        self.src, self.trg, self.mask, self.loc, self.end_num = [], [], [], [], []
        if is_train:
            self.src = x_train
            self.trg = y_train
            self.mask = mask_train
            self.loc = loc_train
            self.end_num = end_num_train  # 记录end_point
        else:
            self.src = x_val
            self.trg = y_val
            self.mask = mask_val
            self.loc = loc_test
            self.end_num = end_num_val

    def __getitem__(self, index):
        return self.src[index], self.trg[index], self.mask[index], self.loc[index], self.end_num[index]

    def __len__(self):
        return len(self.src)


class DLInfMA(nn.Module):
    def __init__(self, time_indim, time_outdim, n_hidden_cat, n_hidden_z, n_hidden_m, n_hidden_p, num_per_batch,
                 d_model, num_heads, num_layers):
        super().__init__()
        self.dense1 = nn.Linear(time_indim, time_outdim)  # (24, h)
        self.dense2 = nn.Linear(n_hidden_cat, n_hidden_z)  # (4+h, z)
        self.dense3 = nn.Linear(1, n_hidden_m)  # (1, m)
        self.dense4 = nn.Linear(n_hidden_m, n_hidden_p)  # (m, p)
        self.dense5 = nn.Linear(n_hidden_z, n_hidden_p)  # (z, p)
        self.bn1 = nn.BatchNorm1d(num_per_batch)  # (num_per_batch)
        self.dense6 = nn.Linear(n_hidden_p, 1)  # (z, p)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()


    def forward(self, data_all):
        data, mask = data_all[0], data_all[1]
        x1 = data[:, :, :3].to(torch.float32)  # (batch, loc_num, 4)
        x2 = data[:, :, 4:].to(torch.float32)  # (batch, loc_num, 24)
        x3 = data[:, :, 3].to(torch.float32)  # (batch, loc_num, 1)
        x3 = torch.unsqueeze(x3, 2)
        x2 = self.dense1(x2)  # (batch, loc_num, h)
        x2 = self.relu(x2)
        x = torch.cat([x1, x2], dim=2)  # (batch, loc_num, h+4)
        x = self.dense2(x)  # (batch, loc_num, z)
        x = self.relu(x)
        x = x.permute(1, 0, 2)  # (loc_num, batch, z)
        x = self.transformer_encoder(x, src_key_padding_mask=mask)  # (loc_num, batch, z)
        x = x.permute(1, 0, 2)  # (batch, loc_num, z)
        x3 = self.dense3(x3)  # (batch, loc_num, m)
        x3 = self.dense4(x3)  # (batch, loc_num, p)
        x = self.dense5(x)  # (batch, loc_num, p)
        x = self.bn1(x)
        x3 = self.bn1(x3)
        x = x + x3
        x = self.tanh(x)  # (batch, loc_num, p)
        x = self.dense6(x)  # (batch, loc_num, 1)
        x = x.reshape(x.shape[0], -1)  # (batch, loc_num)
        out = x*(1-mask)
        # out = self.softmax(x)
        return out


batch_size = 2
data_train = MyDataset(is_train=True)
data_test = MyDataset(is_train=False)
data_loader_train = DataLoader(data_train, batch_size=batch_size, shuffle=False)
data_loader_val = DataLoader(data_test, batch_size=batch_size, shuffle=False)

model = DLInfMA(time_indim=24, time_outdim=5, n_hidden_cat=8, n_hidden_z=128, n_hidden_m=128, n_hidden_p=128,
                num_per_batch=max_len, d_model=128, num_heads=2, num_layers=1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
print(model)

# optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


def train(epoch):
    model.train()
    train_loss = 0
    correct = 0
    sample_num = 0
    for data, label, mask, loc, end_num in data_loader_train:
        if torch.cuda.is_available():
            data = data.cuda()
            label = label.cuda()
        optimizer.zero_grad()
        output = model((data, mask))
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        # max_norm = 100
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm, norm_type=2)
        train_loss += loss.item() * data.size(0)
        prediction = torch.argmax(output, 1)
        correct += (prediction == label.argmax(dim=1)).sum().item()
        sample_num += len(prediction)
    train_loss = train_loss / len(data_loader_train.dataset)
    acc = correct / sample_num
    # print('Epoch: {} \tTraining Loss: {}'.format(epoch, train_loss))
    # print('Epoch: {} \tTraining Loss: {:.6f}, Acc:: {}'.format(epoch, train_loss, acc))
    # for name, parameters in model.named_parameters():
    #     if name == 'dense1.weight':
    #         print(name, ':', parameters)

    return train_loss


def val(epoch):
    model.eval()
    val_loss = 0
    correct = 0
    sample_num = 0
    label_loc_list = []  # 存放真实位置坐标列表
    pred_loc_list = []  # 存放预测位置坐标列表

    with torch.no_grad():
        for data, label, mask, loc, end_num in data_loader_val:
            if torch.cuda.is_available():
                data = data.cuda()
                label = label.cuda()
            output = model((data, mask))
            loss = criterion(output, label)
            val_loss += loss.item() * data.size(0)
            prediction = torch.argmax(output, 1)
            correct += (prediction == label.argmax(dim=1)).sum().item()
            sample_num += len(prediction)
            label_loc = []
            pred_loc = []
            for i, k in zip(range(len(data)), end_num):
                end_point = data_num2end_point[int(k)]
                label_loc.append(end_point_real_loc[end_point])
            for i, j, k in zip(range(len(data)), prediction, end_num):
                end_point = data_num2end_point[int(k)]
                if (loc[i, j, :].numpy() == np.array([0, 0])).all():
                    lab_loc = end_point_lab_loc[end_point]
                    pred_loc.append(np.array(lab_loc))
                else:
                    pred_loc.append(loc[i, j, :].numpy())
            # label_loc = [loc[i, j, :].numpy() for i, j in zip(range(len(data)), label.argmax(dim=1))]
            # pred_loc = [loc[i, j, :].numpy() for i, j in zip(range(len(data)), prediction)]
            label_loc_list.extend(label_loc)
            pred_loc_list.extend(pred_loc)

    val_loss = val_loss / len(data_loader_val.dataset)
    acc = correct / sample_num
    # print('Epoch: {} \tTesting Loss: {:.6f}'.format(epoch, val_loss))
    print('Epoch: {} \tTesting Loss: {:.6f}, Acc:: {}'.format(epoch, val_loss, acc))

    return val_loss, label_loc_list, pred_loc_list


loss_train = []
loss_test = []
epochs = 100
for epoch in range(epochs):
    # for name, parameters in model.named_parameters():
    #     print(name, ':', parameters)
    #     break
    loss_train.append(train(epoch))
    val_loss, label_loc_list, pred_loc_list = val(epoch)
    loss_test.append(val_loss)
    print(metrics(label_loc_list, pred_loc_list))

writer = SummaryWriter('./pytorch_tb1')
for index, (x, y) in enumerate(zip(loss_train, loss_test)):
    writer.add_scalar("x", x, index)  # 日志中记录x在第step i 的值
    writer.add_scalar("y", y, index)  # 日志中记录y在第step i 的值
writer.close()

