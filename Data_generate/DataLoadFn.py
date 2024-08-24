import numpy as np
import torch

from Data_generate.Data_file.IMSdata_dir import T_IMS
from Data_generate.Data_file.CQUdata_dir import T_CQU
from Data_generate.Data_file.OTWdata_dir import T_OTW

from Data_generate.mat2csv import get_data_csv
from my_utils.training_utils import my_normalization

normalization = my_normalization


def sample_shuffle(data):
    """
    required: data.shape [Nc, num, ...]
    :param data: [[Nc, num, ...]]
    """
    for k in range(data.shape[0]):
        np.random.shuffle(data[k])
    return data


class DataGenFn:
    def __init__(self):

        # IMS data:
        self.ims = [T_IMS]

        # CQU data:
        self.cqu = [T_CQU]

        # OTW data:
        self.otw = [T_OTW]

    def IMS_3way(self, way=3, examples=100, split=30, shuffle=False,
                 data_len=2048, normalize=True, label=False):
        """
        :param shuffle:
        :param split:
        :param way: 3/7
        :param label:
        :param examples: examples of each file
        :param data_len: size of each example
        :param normalize: normalize data
        :return: [Nc,split,1,2048], [Nc, split];
        [Nc,examples*2-split,1,2048], [Nc, examples*2-split]

        """
        file_dir = self.ims

        print('IMS_{}way loading ……'.format(way))
        # print(file_dir)
        n_way = len(file_dir[0])  # 3/7 way
        n_file = len(file_dir)  # 2 files
        num_each_file = examples
        num_each_way = examples * n_file
        data_size = num_each_file * data_len
        data_set = None
        for i in range(n_way):
            data_ = np.zeros([n_file, num_each_file, data_len])
            for j in range(n_file):
                data = get_data_csv(file_dir=file_dir[j][i], num=data_size, header=0, shift_step=200)
                data = data.reshape([-1, data_len])
                data_[j] = data
            data_ = data_.reshape([-1, data_len])  # [num_each_way, 2048]
            if normalize:
                data_ = normalization(data_)
            if i == 0:
                data_set = data_
            else:
                data_set = np.concatenate((data_set, data_), axis=0)
        data_set = data_set.reshape([n_way, num_each_way, 1, data_len])
        if shuffle:
            data_set = sample_shuffle(data_set)  # 酌情shuffle, 有的时候需要保持测试集和evaluate一致
        train_data, test_data = data_set[:, :split], data_set[:, split:]  # 先shuffle
        train_data, test_data = torch.from_numpy(train_data), torch.from_numpy(test_data)
        train_data, test_data = train_data.float(), test_data.float()

        if label:
            label = torch.arange(n_way, dtype=torch.long).unsqueeze(1)
            label = label.repeat(1, num_each_way)  # [Nc, num_each_way]
            train_lab, test_lab = label[:, :split], label[:, split:]
            return train_data, train_lab, test_data, test_lab  # [Nc,num_each_way,1,2048], [Nc, 50]
        else:
            return train_data, test_data  # [Nc, num_each_way, 1, 2048]

    def CQU_4way(self, way=4, examples=100, split=30, shuffle=False,
                 data_len=2048, normalize=True, label=False):
        """
        :param shuffle:
        :param split:
        :param way: 3/7
        :param label:
        :param examples: examples of each file
        :param data_len: size of each example
        :param normalize: normalize data
        :return: [Nc,split,1,2048], [Nc, split];
        [Nc,examples*2-split,1,2048], [Nc, examples*2-split]

        """
        file_dir = self.cqu

        print('CQU_{}way loading ……'.format(way))
        # print(file_dir)
        n_way = len(file_dir[0])  # 3/7 way
        n_file = len(file_dir)  # 2 files
        num_each_file = examples
        num_each_way = examples * n_file
        data_size = num_each_file * data_len

        data_set = None
        for i in range(n_way):
            data_ = np.zeros([n_file, num_each_file, data_len])
            for j in range(n_file):
                data = get_data_csv(file_dir=file_dir[j][i], num=data_size, header=0, shift_step=200)
                data = data.reshape([-1, data_len])
                data_[j] = data
            data_ = data_.reshape([-1, data_len])  # [num_each_way, 2048]
            if normalize:
                data_ = normalization(data_)
            if i == 0:
                data_set = data_
            else:
                data_set = np.concatenate((data_set, data_), axis=0)
        data_set = data_set.reshape([n_way, num_each_way, 1, data_len])
        print(data_set.shape)
        if shuffle:
            data_set = sample_shuffle(data_set)  # 酌情shuffle, 有的时候需要保持测试集和evaluate一致
        train_data, test_data = data_set[:, :split], data_set[:, split:]  # 先shuffle
        train_data, test_data = torch.from_numpy(train_data), torch.from_numpy(test_data)
        train_data, test_data = train_data.float(), test_data.float()

        if label:
            label = torch.arange(n_way, dtype=torch.long).unsqueeze(1)
            label = label.repeat(1, num_each_way)  # [Nc, num_each_way]
            train_lab, test_lab = label[:, :split], label[:, split:]
            return train_data, train_lab, test_data, test_lab  # [Nc,num_each_way,1,2048], [Nc, 50]
        else:
            return train_data, test_data  # [Nc, num_each_way, 1, 2048]

    def OTW_3way(self, way=3, examples=100, split=30, shuffle=False,
                 data_len=2048, normalize=True, label=False):
        """
        :param shuffle:
        :param split:
        :param way: 3/7
        :param label:
        :param examples: examples of each file
        :param data_len: size of each example
        :param normalize: normalize data
        :return: [Nc,split,1,2048], [Nc, split];
        [Nc,examples*2-split,1,2048], [Nc, examples*2-split]

        """
        file_dir = self.otw

        print('CQU_{}way loading ……'.format(way))
        # print(file_dir)
        n_way = len(file_dir[0])  # 3/7 way
        n_file = len(file_dir)  # 2 files
        num_each_file = examples
        num_each_way = examples * n_file
        data_size = num_each_file * data_len
        data_set = None
        for i in range(n_way):
            data_ = np.zeros([n_file, num_each_file, data_len])
            for j in range(n_file):
                data = get_data_csv(file_dir=file_dir[j][i], num=data_size, header=0, shift_step=200)
                data = data.reshape([-1, data_len])
                data_[j] = data
            data_ = data_.reshape([-1, data_len])  # [num_each_way, 2048]
            if normalize:
                data_ = normalization(data_)
            if i == 0:
                data_set = data_
            else:
                data_set = np.concatenate((data_set, data_), axis=0)
        data_set = data_set.reshape([n_way, num_each_way, 1, data_len])
        if shuffle:
            data_set = sample_shuffle(data_set)  # 酌情shuffle, 有的时候需要保持测试集和evaluate一致
        train_data, test_data = data_set[:, :split], data_set[:, split:]  # 先shuffle
        train_data, test_data = torch.from_numpy(train_data), torch.from_numpy(test_data)
        train_data, test_data = train_data.float(), test_data.float()

        if label:
            label = torch.arange(n_way, dtype=torch.long).unsqueeze(1)
            label = label.repeat(1, num_each_way)  # [Nc, num_each_way]
            train_lab, test_lab = label[:, :split], label[:, split:]

            return train_data, train_lab, test_data, test_lab  # [Nc,num_each_way,1,2048], [Nc, 50]
        else:

            return train_data, test_data  # [Nc, num_each_way, 1, 2048]


 # 采用滑窗采样的方法 num =  (N-length) // win_size + 1


if __name__ == "__main__":
    d = DataGenFn()

    tr_d, tr_l, te_d, te_l = d.OTW_3way(label=True, way=3, normalize=False, data_len=1024,
                                        examples=200, split=20)
    tr_d, tr_l, te_d, te_l = d.CQU_4way(label=True, way=3, normalize=False, data_len=1024,
                                        examples=200, split=20)
    tr_d, tr_l, te_d, te_l = d.IMS_3way(label=True, way=3, normalize=False, data_len=1024,
                                        examples=200, split=20)

    print(tr_d.shape, tr_l.shape)
    print(tr_d[2, 0, 0, :10], tr_l[:, :3])

