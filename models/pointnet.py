import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

"""
STN: Spatial Transformer Networks  空间转换网络 
就是 T - net
"""


class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        # in_channels: int, out_channels: int, kernel_size: _size_1_t
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]  # 第一个维度是batch的数量
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)  # 转换为列为1024但行不变的数据

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)  # 生成3x3的单位矩阵，但是是一行的形式方便计算
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden  # 这边加起来是什么意思？为什么要加单位矩阵，这边是对应论文说初始化为对角单位阵
        x = x.view(-1, 3, 3)  # 转换为3x3的矩阵
        return x


# 高维映射网络，即将单个点云点映射到多维空间的网络，以避免后续的最大池化过度地损失信息
class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


# 整体pointnet网络结构
class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(channel)  # 3维空间转换矩阵（输入转换）
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat  # 全局特征标志（判断是用于分类任务还是语义分割任务）
        self.feature_transform = feature_transform  # 是否对高维特征进行旋转变换标定
        if self.feature_transform:
            self.fstn = STNkd(k=64)  # 高维空间变换矩阵

    def forward(self, x):
        # B:样本的一个批次大小,batch；D：点的维度 3 (x,y,z) dim ; N:点的数量 (1024) number
        # 即这边一次输入24个样本，一个样本含有1024个点云点， 一个点云点为3维(x,y,z)
        B, D, N = x.size()  # [24, 3, 1024]
        trans = self.stn(x)  # 得到3维旋转转换矩阵
        x = x.transpose(2, 1)  # 将2轴和1轴对调，相当于[24,1024,3]，为了和后面T-net训练出来的旋转矩阵相乘做处理，后两个维度要满足矩阵乘法规则
        if D > 3:  # 这边是是特征点的话，不只有3维(x,y,z)，可能为多维
            x, feature = x.split(3, dim=2)  # 从维度2上按照长度3分开。前3为xyz特征（pointnet必须的），3之后就是其他特征
        x = torch.bmm(x, trans)  # 将3维点云数据进行旋转变换
        if D > 3:
            x = torch.cat([x, feature], dim=2)  # 将其他特征重新放回x中
        x = x.transpose(2, 1)  # 将2轴和1轴再对调，返回原始数据
        x = F.relu(self.bn1(self.conv1(x)))  # 进行第一次卷积、标准化、激活、得到64维的数据

        # 下面是第二层卷积层处理
        if self.feature_transform:  # 如果需要对高维特征进行旋转对齐的话
            trans_feat = self.fstn(x)  # 得到特征空间的旋转矩阵
            x = x.transpose(2, 1)  # 将1轴和2轴对调，为了和后面T-net训练出来的旋转矩阵相乘做处理，后两个维度要满足矩阵乘法规则
            x = torch.bmm(x, trans_feat)  # 将特征数据进行旋转转换
            x = x.transpose(2, 1)  # 将2轴再次和1轴对调恢复原来数据顺序
        else:
            trans_feat = None

        pointfeat = x  # 旋转矫正过后的特征
        x = F.relu(self.bn2(self.conv2(x)))  # 第二次卷积   输出维128
        x = self.bn3(self.conv3(x))  # 第三次卷积 输出维1024
        x = torch.max(x, 2, keepdim=True)[0]  # 进行最大池化处理，只返回最大的数，不返回索引([0]是数值，[1]是索引)
        x = x.view(-1, 1024)  # 把x reshape为 1024列的行数不定矩阵，这边的-1指的就是行数不定。

        if self.global_feat:  # 是否为全局特征（判断是用于分类任务还是语义分割任务）
            return x, trans, trans_feat  # 返回特征数据x，3维旋转矩阵，多维旋转矩阵
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)  # 多扩展了一个维度是为了和局部特征统一维度，方便后面的连接，然后复制成与局部特征一样的数量
            return torch.cat([x, pointfeat], 1), trans, trans_feat  # 这边对应点云分割算法中，将全局特征与局部特征连接。


"""这边是高维特征空间转换举证的正则项，大致的意思的把这个转换矩阵乘上其转置阵再减去单位阵，取剩下差值的均值为损失函数"""


def feature_transform_reguliarzer(trans):
    d = trans.size()[1]  # 矩阵维度
    I = torch.eye(d)[None, :, :]  # 生成同维度的对角单位阵
    if trans.is_cuda:  # 是否采用Cuda加速
        I = I.cuda()

    # 损失函数，将变换矩阵乘自身转置然后减单位阵，取结果的元素均值为损失函数，因为正交阵乘其转置为单位阵。
    # A*(A'-I) = A*A'- A*I = I - A*I |  A’: 矩阵A的转置。TODO:???是不是写错了
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1) - I), dim=(1, 2)))
    return loss
