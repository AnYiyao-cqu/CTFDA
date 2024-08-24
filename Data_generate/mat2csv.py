import csv
import numpy as np
import pandas as pd
import scipy
from scipy import io
import os


def get_filename(file_dir):
    """
    :param file_dir:
    """
    file_name = os.listdir(file_dir)
    if len(file_name) != 1:
        print('===========!!!===========!!!===========')
        print('There are {} files in [{}]'.format(len(file_name), file_dir))
        print(file_name)
        new_file = None
        exit()
    else:
        new_file = os.path.join(file_dir, file_name[-1])
    return new_file


def add_csv(file_dir):
    """

    :param file_dir:
    """
    split_file = os.path.split(file_dir)
    outputFile = os.path.join(split_file[0], split_file[1][:-4] + '.csv')
    return outputFile


def mat2csv_sa(file_dir, name='H_D_DE', channel=4):  # for SA
    """

    :param channel:
    :param file_dir:
    :param name:
    """
    mat_file = scipy.io.loadmat(file_dir)
    name_list = list(mat_file.keys())
    print(name_list)
    outputFile = add_csv(file_dir)
    if name in name_list:
        data = mat_file[name]
        index = channel - 1
        data = pd.DataFrame(data)[index]
        data.to_csv(outputFile, header=0, index=False)
        print('oooooooooooooooooooooooooooooooooooooo')
        print('Transform the file to csv format at: \n', outputFile)


def mat2csv_cw(file_dir, name='DE_time'):  # for CW data
    """

    :param file_dir:
    :param name:
    """
    mat_file = scipy.io.loadmat(file_dir)
    name_list = list(mat_file.keys())
    print(name_list)
    outputFile = add_csv(file_dir)
    name_new = []
    for n in name_list:
        if name in n:
            name_new.append(n)
    if len(name_new) > 1:
        print("More than 1 file named {}:\n {}".format(name, name_new))
        exit()
    else:
        print(name_new)

    data = mat_file[name_new[0]]
    # print(data[:5])
    index = 0
    data = pd.DataFrame(data)[index]
    data.to_csv(outputFile, header=0, index=False)
    print('oooooooooooooooooooooooooooooooooooooo')
    print('Transform the file to csv format at: \n', outputFile)


def get_data_csv(file_dir, num=100000, header=0, shift_step=200):
    """
    :param shift_step:
    :param num:
    :param header:
    :param file_dir:
    """
    data = pd.read_csv(file_dir, header=header).values.reshape(-1)  # DataFrame ==> array
    while data.shape[0] < num:
        # print("Need: {}, Got from a file [{}]: {}".format(num, file_dir, data.shape[0]))
        # print('We implement the operation with Shift_step: {}\n'.format(shift_step))
        header = header + shift_step
        data_ = pd.read_csv(file_dir, header=header).values.reshape(-1)
        data = np.concatenate((data, data_), axis=0)
    data = data[:num]
    # data = np.transpose(data, axes=[1, 0]).reshape(-1)
    return data


if __name__ == "__main__":

    file_ = r'E:\CTFDA\data\Ottawa\H_D.mat'
    mat2csv_sa(file_, channel=2)