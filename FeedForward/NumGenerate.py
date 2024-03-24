import numpy
import torch
from torch.utils.data import TensorDataset, DataLoader

def Generate_dataset(Num_SampleSize):
    x = numpy.linspace(Num_SampleSize)
    y = numpy.log2(x) + numpy.cos(numpy.pi * x / 2)
    return torch.tensor(x).view(-1, 1), torch.tensor(y).view(-1, 1)

def Data_Process(Num_SampleSize, setting_batch_size):
    x, y = Generate_dataset(Num_SampleSize)

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
