#!/usr/bin/env python
# coding: utf-8

import os
import re
import itertools
import numpy as np
import pandas as pd
import scipy.io as scio
import torch
from scipy import signal
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

np.random.seed(2048)
torch.manual_seed(2048)
torch.cuda.manual_seed_all(2048)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())
print(torch.cuda.get_device_name(0))

# dim = 2048
# type_list = ['outer', 'cage', 'outer&inner', 'inner', 'ball']
# bs = 64
# Epoch = 100
# # vis = visdom.Visdom(env='model_extractor')
# winner = False


raw_num = 240
dim = col_num = 2000
batch_size = 64

cwru_root = '../Data/'
xjtu_root = '../XJTU-SY_Bearing_Datasets/'


class cwru_data(object):
    # inner: 0, outer: 1, ball:2, normal: 3
    # 'IR021_3.mat': 0, 'IR007_3.mat': 0, 'OR007@6_3.mat': 1, 'B021_3.mat': 3,
    #  'IR014_3.mat': 0, 'OR021@6_3.mat': 1, 'OR014@6_3.mat': 1,
    #  'B014_3.mat': 3, 'Normal_3.mat': 2, 'B007_3.mat': 3

    # (label) 0,1,2: (num)720, 720, 240

    # full: whether remove 'ball' fault files: True -> not remove;
    def __init__(self, full=True):
        self.full = full
        self.name = self.file_list()
        self.data = self.get_data()  # 2400x2000
        self.label = self.get_label()  # 2400,   1-d

    def file_list(self):
        ls = os.listdir(cwru_root)
        if not self.full:
            forbidden_file = ['B021_3.mat', 'B014_3.mat', 'B007_3.mat', 'Normal_3.mat']
            for i in forbidden_file:
                ls.remove(i)
            # print(ls)
        return ls

    def get_data(self):
        for i in range(len(self.name)):
            file = scio.loadmat(cwru_root + self.name[i])
            for k in file.keys():
                file_matched = re.match('X\d{3}_DE_time', k)  # DE - 驱动端加速度数据
                if file_matched:
                    key = file_matched.group()
            if i == 0:
                #                 print(file[key].shape) 489125
                data = np.array(file[key][0:480000].reshape(raw_num, col_num))  # 把每个数据集中的数据划分为240份
            else:
                data = np.vstack((data, file[key][0:480000].reshape((raw_num, col_num))))
        print("CWRU shape:")
        print(data.shape)
        return data

    def get_label(self):
        # title = np.array([i.replace('.mat', '') for i in self.name])
        name_map = {'IR021_3.mat': 0, 'IR007_3.mat': 0, 'OR007@6_3.mat': 1, 'B021_3.mat': 3,
                    'IR014_3.mat': 0, 'OR021@6_3.mat': 1, 'OR014@6_3.mat': 1,
                    'B014_3.mat': 3, 'Normal_3.mat': 2, 'B007_3.mat': 3}
        label = [name_map[i] for i in self.name]
        label_copy = np.copy(label)
        for _ in range(raw_num - 1):
            label = np.vstack((label, label_copy))
        return label.T.flatten()


class xjtu_data(object):
    # inner: 0, outer: 1, ball:2, normal: 3
    folder_35hz = xjtu_root + '35Hz12kN/'
    folder_37hz = xjtu_root + '37.5Hz11kN/'
    folder_40hz = xjtu_root + '40Hz10kN/'
    type_list = ['inner', 'outer', 'normal', 'ball', 'cage', 'outer&inner']
    data = []
    label = []
    resized_label_set = []
    resized_data_set = []

    def __init__(self, full=True):
        self.full = full
        self.init_data()

    def init_data(self):
        root_t = self.folder_37hz
        target_data = {
            "Bearing1_1": {"I": 78, "II": 88, "III": 123, 'type': 'outer'},
            "Bearing1_2": {"I": 55, "II": 75, "III": 161, 'type': 'outer'},
            "Bearing1_3": {"I": 58, "II": 130, "III": 158, 'type': 'outer'},
            "Bearing1_4": {"I": 74, "II": 90, "III": 122, 'type': 'cage'},
            "Bearing1_5": {"I": 35, "II": 41, "III": 52, 'type': 'outer&inner'},
            "Bearing2_1": {"I": 452, "II": 454, "III": 491, 'type': 'inner'},
            "Bearing2_2": {"I": 50, "II": 51, "III": 161, 'type': 'outer'},
            "Bearing2_4": {"I": 30, "II": 31, "III": 42, 'type': 'outer'},
            "Bearing2_5": {"I": 120, "II": 121, "III": 339, 'type': 'outer'},
            "Bearing3_1": {"I": 2463, "II": 2464, "III": 2536, 'type': 'outer'},
            "Bearing3_3": {"I": 340, "II": 341, "III": 369, 'type': 'inner'},
            "Bearing3_4": {"I": 1416, "II": 1417, "III": 1514, 'type': 'inner'},
            "Bearing3_5": {"I": 10, "II": 11, "III": 110, 'type': 'outer'},
        }
        enabled_sets = ["Bearing2_1", "Bearing2_2", "Bearing2_4", "Bearing2_5"]
        target_set = np.array([])
        target_label_set = np.array([])
        for key in enabled_sets:
            hvs_set = np.array([])
            # hvs_normal_set = np.array([])
            for i in range(target_data[key]["II"], target_data[key]["III"]):
                data = pd.read_csv(root_t + f"{key}/{i + 1}.csv")
                hvs = np.array(data["Horizontal_vibration_signals"])
                hvs_set = np.hstack((hvs_set, hvs))

            # for j in range(0, target_data[key]["I"]):
            #     data = pd.read_csv(root_t + f"{key}/{j + 1}.csv")
            #     hvs_normal = np.array(data["Horizontal_vibration_signals"])
            #     hvs_normal_set = np.hstack((hvs_normal_set, hvs_normal))

            if target_set.size == 0:
                target_set = np.array([hvs_set[i * dim:i * dim + dim] for i in range(hvs_set.shape[0] // dim)])
                size = target_set.shape[0]
                target_label_set = np.array([self.type_list.index(target_data[key]['type'])] * size)

                # target_set = np.vstack(
                #     (target_set,
                #      np.array([hvs_normal_set[i * dim:i * dim + dim] for i in range(hvs_normal_set.shape[0] // dim)])))
                # size = target_set.shape[0] - size
                # target_label_set = np.hstack(
                #     (target_label_set, np.array([self.type_list.index('normal')] * size)))

            else:
                size0 = target_set.shape[0]
                target_set = np.vstack(
                    (target_set, np.array([hvs_set[i * dim:i * dim + dim] for i in range(hvs_set.shape[0] // dim)])))
                size = target_set.shape[0] - size0
                target_label_set = np.hstack(
                    (target_label_set, np.array([self.type_list.index(target_data[key]['type'])] * size)))

                # size0 = target_set.shape[0]
                # target_set = np.vstack(
                #     (target_set,
                #      np.array([hvs_normal_set[i * dim:i * dim + dim] for i in range(hvs_normal_set.shape[0] // dim)])))
                # size = target_set.shape[0] - size0
                # target_label_set = np.hstack(
                #     (target_label_set, np.array([self.type_list.index('normal')] * size)))
        self.label = target_label_set
        self.data = target_set
        label_zeros = np.where(target_label_set == 0)
        label_zeros_list = label_zeros[0]
        print(f'zeros:{len(label_zeros[0])}')
        label_ones = np.where(target_label_set == 1)
        label_ones_list = label_ones[0]
        print(f'ones:{len(label_ones[0])}')
        # label_twos = np.where(target_label_set == 2)
        # label_twos_list = label_twos[0]
        # print(f'twos:{len(label_twos[0])}')

        resized_label_set = target_label_set[label_zeros_list[0]]
        resized_data_set = target_set[label_zeros_list[0]]

        # for i in range(1, len(label_zeros_list)):
        for i in range(1, 491):
            resized_label_set = np.hstack((resized_label_set, target_label_set[label_zeros_list[i]]))
            resized_data_set = np.vstack((resized_data_set, target_set[label_zeros_list[i]]))
        for i in range(0, 491):
            resized_label_set = np.hstack((resized_label_set, target_label_set[label_ones_list[i]]))
            resized_data_set = np.vstack((resized_data_set, target_set[label_ones_list[i]]))
            # resized_label_set = np.hstack((resized_label_set, target_label_set[label_twos_list[i]]))
            # resized_data_set = np.vstack((resized_data_set, target_set[label_twos_list[i]]))
        # for i in range(0, 240):
        #     resized_label_set = np.hstack((resized_label_set, target_label_set[label_ones_list[i]]))
        #     resized_data_set = np.vstack((resized_data_set, target_set[label_ones_list[i]]))
        # count0 = 0
        # count1 = 0
        # count2 = 0
        # for hh in resized_label_set:
        #     if hh == 0:
        #         count0 += 1
        #     elif hh == 1:
        #         count1 += 1
        #     elif hh == 2:
        #         count2 += 1
        # print(count0)
        # print(count1)
        # print(count2)
        #
        self.resized_data_set = resized_data_set
        self.resized_label_set = resized_label_set
        return resized_label_set


class MyDataset(Dataset):
    def __init__(self, data, label):
        super().__init__()
        self.data = data.reshape(data.shape[0], 1, 1000)
        self.label = label
        self.length = data.shape[0]

    def __getitem__(self, index):
        hdct = self.data[index, :, :]  # 读取每一个npy的数据
        ldct = self.label[index]
        return hdct, ldct  # 返回数据还有标签

    def __len__(self):
        return self.length  # 返回数据的总个数


def load(batch_size, kwargs):
    source = cwru_data(full=False)
    source_data = source.data
    source_label = source.label

    target = xjtu_data(full=False)
    target_data = target.resized_data_set
    # target_data = target.data
    target_label = target.resized_label_set
    # target_label = target.label

    #stft: target data
    # fs = 25.6e3
    # target_stft_set = np.array(
    #     [signal.stft(target_data[i, :], fs, nperseg=128)[2] for i in range(target_data.shape[0])])
    # target_stft_set = target_stft_set.reshape(target_stft_set.shape[0], 1, target_stft_set.shape[1],
    #                                           target_stft_set.shape[2])

    #downsampling
    index = np.arange(0, 2000, 2)
    source_data = source_data[:, index]
    target_data = target_data[:, index]

    #
    source_data = np.save('source_data.npy', source_data)
    source_label = np.save('source_label.npy', source_label)
    target_data = np.save('target_data.npy', target_data)
    target_label = np.save('target_label.npy', target_label)

    source_data = np.load('source_data.npy')
    source_label = np.load('source_label.npy')
    target_data = np.load('target_data.npy')
    target_label = np.load('target_label.npy')

    print(target_data.shape)
    print(target_label.shape)
    print(source_data.shape)
    print(source_label.shape)


    # print(target_stft_set.shape)

    # stft: source data
    # fs = 12e3
    # source_stft_set = np.array(
    #     [signal.stft(source_data[i, :], fs, nperseg=128)[2] for i in range(source_data.shape[0])])
    # source_stft_set = source_stft_set.reshape(source_stft_set.shape[0], 1, source_stft_set.shape[1],
    #                                           source_stft_set.shape[2])
    # print(source_stft_set.shape)

    # split test set
    # tar_train, tar_test, tar_label_train, tar_label_test = train_test_split(
    #     target_stft_set, target_label, test_size=0.3, random_state=42, stratify=target_label)

    source_dataset = MyDataset(source_data, source_label)
    source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=False, drop_last=True, **kwargs)
    target_dataset = MyDataset(target_data, target_label)
    target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=False, drop_last=True, **kwargs)

    target_dataset_test = MyDataset(target_data, target_label)
    target_loader_test = DataLoader(target_dataset_test, batch_size=batch_size, shuffle=False, drop_last=False,
                                    **kwargs)
    return source_loader, target_loader, target_loader_test


source_loader, target_loader, test_loader = load(32, {})


for i,j in source_loader:
     print(j)


class Residual(nn.Module):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv1d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv1d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm1d(num_channels)
        self.bn2 = nn.BatchNorm1d(num_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


class FeatureExtractor(nn.Module):

    def __init__(self):
        super(FeatureExtractor, self).__init__()
        #
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(1, 20, kernel_size=5, stride=1), #65-4 *33-4 = 61*29
        #     nn.BatchNorm2d(20),
        #     nn.MaxPool2d(2), #61/2 * 29/2 = 30*14
        #     nn.ReLU()
        # )

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 30, kernel_size=20, stride=2),  # (250-20)/2+1 = 116
            nn.BatchNorm1d(30),
            nn.MaxPool1d(2),  # 116/2 = 58
            nn.ReLU()
        )

        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(20, 50, kernel_size=5, stride=1), #30-4 * 14-4 = 26*10
        #     nn.BatchNorm2d(50),
        #     nn.MaxPool2d(2), #26/2 * 10/2 = 13*5
        #     nn.ReLU()
        # )
        self.conv2 = nn.Sequential(
            nn.Conv1d(30, 50, kernel_size=10, stride=2),  # (58-10)/2+1 = 25
            nn.BatchNorm1d(50),
            nn.MaxPool1d(2),  # 25/2=12
            nn.ReLU()
        )

        self.residualBlock = nn.Sequential(Residual(50, 128, use_1x1conv=True),
                                           Residual(128, 256, use_1x1conv=True),
                                           nn.AdaptiveAvgPool1d((1)),
                                           nn.Flatten())


        # self.fc1 = nn.Sequential(
        #     nn.Linear(1620, 810),  # 30*63 = 1890
        #     nn.BatchNorm1d(810),
        #     nn.ReLU()
        # )
        self.dpout = nn.Dropout(0.5)

    def forward(self, x):
        print(x.shape)
        out = self.conv1(x)
        # self.dpout(out)
        out = self.conv2(out)
        out=self.residualBlock(out)
        out = out.view(out.shape[0], -1)
        # out = self.fc1(out)
        return out


# if __name__ == '__main__':
#     model = FeatureExtractor()
#     input = torch.randn(2, 1, 250)  # bz, channel, frequency, segment
#     print(model(input).shape)


class LabelPredictor(nn.Module):

    def __init__(self):
        super(LabelPredictor, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        out = self.fc1(x)
        #         out = out.view(out.shape[0], -1)
        return out


class DomainClassifier(nn.Module):

    def __init__(self):
        super(DomainClassifier, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.fc3 = nn.Sequential(
            nn.Linear(64, 1),
            #             nn.Softmax(dim=1)
        )

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        out = out.view(out.shape[0], -1)
        return out


# input = torch.randn(2, 500)  # bz, features
# model = discriminator()
# print(model(input).shape)


feature_extractor = FeatureExtractor().cuda()
label_predictor = LabelPredictor().cuda()
domain_classifier = DomainClassifier().cuda()

class_criterion = nn.CrossEntropyLoss()
domain_criterion = nn.BCEWithLogitsLoss()

optimizer_F = optim.Adam(feature_extractor.parameters())
optimizer_C = optim.Adam(label_predictor.parameters())
optimizer_D = optim.Adam(domain_classifier.parameters())


def train_epoch(source_dataloader, target_dataloader, current):
    label_predictor.train()
    feature_extractor.train()
    '''
      Args:
        source_dataloader: source data的dataloader
        target_dataloader: target data的dataloader
        lamb: control the balance of domain adaptatoin and classification.
    '''

    # D loss: Domain Classifier的loss
    # F loss: Feature Extrator & Label Predictor的loss
    running_D_loss, running_F_loss = 0.0, 0.0
    total_hit, total_num = 0.0, 0.0

    lamb = 2./(1.+np.exp(-10*current))-1


    for i, ((source_data, source_label), (target_data, _)) in enumerate(zip(source_dataloader, target_dataloader)):
        source_data = source_data.cuda()
        source_label = source_label.long().cuda()
        target_data = target_data.cuda()

        # Mixed the source data and target data, or it'll mislead the running params
        #   of batch_norm. (runnning mean/var of soucre and target data are different.)
        mixed_data = torch.cat([source_data, target_data], dim=0)
        mixed_data = mixed_data.type(torch.FloatTensor)
        mixed_data = mixed_data.cuda()
        #         print(mixed_data.shape)
        domain_label = torch.zeros([source_data.shape[0] + target_data.shape[0], 1]).cuda()
        # set domain label of source data to be 1.
        domain_label[:source_data.shape[0]] = 1

        # Step 1 : train domain classifier
        feature = feature_extractor(mixed_data)
        # We don't need to train feature extractor in step 1.
        # Thus we detach the feature neuron to avoid backpropgation.
        domain_logits = domain_classifier(feature.detach())
        loss = domain_criterion(domain_logits, domain_label)
        running_D_loss += loss.item()
        loss.backward()
        optimizer_D.step()

        # Step 2 : train feature extractor and label classifier
        class_logits = label_predictor(feature[:source_data.shape[0]])
        domain_logits = domain_classifier(feature)
        # loss = cross entropy of classification - lamb * domain binary cross entropy.
        #  The reason why using subtraction is similar to generator loss in disciminator of GAN
        loss = class_criterion(class_logits, source_label) - lamb * domain_criterion(domain_logits, domain_label)
        running_F_loss += loss.item()
        loss.backward()
        optimizer_F.step()
        optimizer_C.step()

        optimizer_D.zero_grad()
        optimizer_F.zero_grad()
        optimizer_C.zero_grad()

        total_hit += torch.sum(torch.argmax(class_logits, dim=1) == source_label).item()
        total_num += source_data.shape[0]
        print(i, end='\r')

    return running_D_loss / (i + 1), running_F_loss / (i + 1), total_hit / total_num


# draw confusion matrix
def plot_confusion_matrix(cm, classes, normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def testing():
    result = []
    first = 0
    label_predictor.eval()
    feature_extractor.eval()
    total_hit, total_num = 0.0, 0.0
    for i, (test_data, test_label) in enumerate(test_loader):
        test_data = test_data.type(torch.FloatTensor)
        test_data = test_data.cuda()
        test_label = test_label.cuda()
        class_logits = label_predictor(feature_extractor(test_data))
        total_hit += torch.sum(torch.argmax(class_logits, dim=1) == test_label).item()
        total_num += test_data.shape[0]
    print('target domain test accuracy: {:6.4f}'.format(total_hit / total_num))

# train 200 epochs
for epoch in range(200):
    # You should chooose lamnda cleverly.
    train_D_loss, train_F_loss, train_acc = train_epoch(source_loader, target_loader, current=epoch)
    testing()

    # torch.save(feature_extractor.state_dict(), f'extractor_model.bin')
    # torch.save(label_predictor.state_dict(), f'predictor_model.bin')

    print('epoch {:>3d}: train D loss: {:6.4f}, train F loss: {:6.4f}, acc {:6.4f}'.format(epoch, train_D_loss,
                                                                                           train_F_loss, train_acc))



result = []
first=0
label_predictor.eval()
feature_extractor.eval()
total_hit, total_num = 0.0, 0.0
for i, (test_data, test_label) in enumerate(test_loader):
    test_data = test_data.type(torch.FloatTensor)
    test_data = test_data.cuda()
    test_label = test_label.cuda()
    class_logits = label_predictor(feature_extractor(test_data))
    if first == 0:
        outputSum = class_logits
        targetSum = test_label
        first += 1
    else:
        outputSum = torch.cat([outputSum, class_logits], dim=0)
        targetSum = torch.cat([targetSum, test_label], dim=0)
        first += 1
    # print(test_label)
    # print(torch.argmax(class_logits, dim=1))
    total_hit += torch.sum(torch.argmax(class_logits, dim=1) == test_label).item()
    total_num += test_data.shape[0]
print('target domain test accuracy: {:6.4f}'.format(total_hit / total_num))

pred_classes = torch.argmax(outputSum, dim=1)
confusion_mtx = confusion_matrix(targetSum.cpu(), pred_classes.cpu())
plot_confusion_matrix(confusion_mtx, classes=range(2))
# print(feature_output.grad.shape)


# PATH = "./model/experiment_extractor/"
# # #保存
# torch.save(src_feature_extractor, PATH+"src_feature_extractor8.pt")
# torch.save(tar_feature_extractor, PATH+"tar_feature_extractor8.pt")
# torch.save(D, PATH+"D8.pt")
# torch.save(C, PATH+"C8.pt")


'''
'''
