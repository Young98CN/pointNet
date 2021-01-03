import numpy as np
import warnings
import os
from torch.utils.data import Dataset

# 去除运行过程中的warning
warnings.filterwarnings('ignore')


def pc_normalize(pc):  # 简单的标准化
    centroid = np.mean(pc, axis=0)  # 计算簇的中心点，新的中心点每一个特征的值，是该簇所有数据在该特征的平均值
    pc = pc - centroid  # 去偏移，将坐标系原点转换到形心位置，坐标系只平移不旋转
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))  # 取最大距离，将同一行的元素取平方相加，再开方，取最大。  sqrt(x^2+y^2+z^2)
    pc = pc / m  # 归一化,归一化操作似乎会丢失物品的尺寸大小信息？  因为每个样本的m不一样。
    return pc


# 原来1w个点， 你现在只提取1024个点，FPS就是希望你提取的这1024个点能够近可能的表示点云的初始信息
def farthest_point_sample(point, npoint):  # 最远点的提取
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    # 先随机初始化一个centroids矩阵，
    # 后面用于存储npoint个采样点的索引位置
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))  # 重心
    # 利用distance矩阵记录某个样本中所有点到某一个点的距离
    distance = np.ones((N,)) * 1e10  # # 初值给个比较大的值，后面会迭代更新
    # 利用farthest表示当前最远的点，也是随机初始化，范围为0~N
    farthest = np.random.randint(0, N)
    # 直到采样点达到npoint，否则进行如下迭代
    for i in range(npoint):
        # 设当前的采样点centroids为当前的最远点farthest；
        centroids[i] = farthest
        # 取出这个中心点centroid的坐标
        centroid = xyz[farthest, :]
        # 求出所有点到这个farthest点的欧式距离，存在dist矩阵中
        dist = np.sum((xyz - centroid) ** 2, -1)
        # 建立一个mask，如果dist中的元素小于distance矩阵中保存的距离值，
        # 则更新distance中的对应值，
        # 即记录某个样本中每个点距离所有已出现的采样点的最小距离
        mask = dist < distance
        distance[mask] = dist[mask]
        # 最后从distance矩阵取出最远的点为farthest，继续下一轮迭代
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    # 返回结果是npoint个采样点在原始点云中的索引
    return point


#  制作这个类的重点在于生成一个列表，这个列表的元素为(path_sample x,lable x)的形式，重要的是生成路径与标签的列表
#  也不一定需要制作路径的列表，可能制作路径的列表会比较不占内存，每次只把需要的数据加载进来而已。
#  可以直接制作数据与标签的列表，可以按索引进行连接。
class ModelNetDataLoader(Dataset):  # 自己的数据集类子类，需要集成父类  Dataset
    def __init__(self, root, npoint=1024, split='train', uniform=False, normal_channel=True, cache_size=15000):
        self.root = root  # 根目录
        self.npoints = npoint  # 对原始数据集下采样至1024个点
        self.uniform = uniform  # 是否归一化
        self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')  # cat  数据库类别的路径 （40类）
        # rstrip删除右边的空格，在for循环中执行line.rstrip()
        # 读取所有种类，list保存，为了用zip打包
        self.cat = [line.rstrip() for line in open(self.catfile)]  # 打开txt文件,读取每一行，并用rstrip（）删除每行末尾的空格
        # dict创建字典,zip把样本和标签一一对应起来（range的返回值也只字典，zip打包list）
        self.classes = dict(zip(self.cat, range(len(self.cat))))  # 生成类别字典{类别1：0,类别2：1,...类别40：39}
        self.normal_channel = normal_channel  # 是否为标准的通道？标准通道指的是什么

        # 把训练集和测试集存入字典
        shape_ids = {}
        # 将数据集划分为训练集和测试集
        # key对应一个list
        shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
        shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        # 断言assert，若split不是'train'或者'test'则报错
        assert (split == 'train' or split == 'test')
        # 对字典中的train或者test对应的value数组进行遍历，join字符串拼接,将类别存储在list中
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        # list of (shape_name, shape_txt_file_path) tuple
        # [(shape_name, shape_txt_file_path)] 元组组成的列表
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print('The size of %s data is %d' % (split, len(self.datapath)))

        # 在内存中缓存数据点的大小
        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = {}  # from index to (point_set, cls) tuple

    def _get_item(self, index):  # 这边的任务主要是写好读取一个样本例子的示范代码，包括数据的初步预处理，如对齐什么的，返回一个（样本，label）,
        # 这边通过开辟了一个缓存区，使用的数据的时候先判断在不在缓存区里面，如果在则直接使用，不再的话再初始化载入，按理说也可以直接把所有的数据一次性加载进来，然后按照索引读取。
        if index in self.cache:
            point_set, cls = self.cache[index]
        else:
            fn = self.datapath[index]  # 不大理解这边的意思
            cls = self.classes[self.datapath[index][0]]
            cls = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(
                np.float32)  # 样本矩阵为（10000，6）格式表示每个样本具有10000个点云点，每一列的意义为x,y,z,r,g,b
            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints, :]

            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])  # 将x，y，z 标准化

            if not self.normal_channel:  # 应该是说不需要彩色信息的时候，只取前面的3列数据
                point_set = point_set[:, 0:3]

            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls)

        return point_set, cls

    # 自定义数据集类子类必须要有重载两个方法，否则会报错
    def __len__(self):
        return len(self.datapath)

    def __getitem__(self, index):
        return self._get_item(index)


if __name__ == '__main__':
    import torch

    data = ModelNetDataLoader('/data/modelnet40_normal_resampled/', split='train', uniform=False, normal_channel=True)
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for point, label in DataLoader:
        print(point.shape)
        print(label.shape)
