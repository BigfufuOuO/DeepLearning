import numpy
import torch
from torch.utils.data import TensorDataset, DataLoader


Num_SampleSize = 200


def Generate_dataset(Num_SampleSize):
    x = numpy.linspace(1, 16, Num_SampleSize)
    y = numpy.log2(x) + numpy.cos(numpy.pi * x / 2)
    return x, y

def Data_Process(Num_SampleSize, setting_batch_size, x_file, y_file):
    x = numpy.load(x_file)
    y = numpy.load(y_file)
    x = torch.tensor(x).view(-1, 1)
    y = torch.tensor(y).view(-1, 1)

    # 划分
    set = torch.randperm(Num_SampleSize)
    train_set = set[:int(Num_SampleSize*0.8)]
    val_set = set[int(Num_SampleSize*0.8):int(Num_SampleSize*0.9)]
    test_set = set[int(Num_SampleSize*0.9):]

    x_train, y_train = x[train_set], y[train_set]
    x_val, y_val = x[val_set], y[val_set]
    x_test, y_test = x[test_set], y[test_set]

    # 转化为Dataset
    train_Dataset = TensorDataset(x_train, y_train)
    val_Dataset = TensorDataset(x_val, y_val)
    test_Dataset = TensorDataset(x_test, y_test)

    # 加载数据
    train_load = DataLoader(train_Dataset, batch_size=setting_batch_size)
    val_load = DataLoader(val_Dataset, batch_size=setting_batch_size)
    test_load = DataLoader(test_Dataset, batch_size=setting_batch_size)

    return train_load, val_load, test_load

if __name__ == "__main__":
    x_filename = f'data/data_x_N={Num_SampleSize}.npy'
    y_filename = f'data/data_y_N={Num_SampleSize}.npy'
    x, y = Generate_dataset(Num_SampleSize)
    numpy.save(x_filename, x)
    numpy.save(y_filename, y)