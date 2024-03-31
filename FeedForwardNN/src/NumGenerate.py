import numpy
import torch
import os
from torch.utils.data import TensorDataset, DataLoader


Num_SampleSize = 200



def Generate_dataset(Num_SampleSize):
    x = numpy.linspace(1, 16, Num_SampleSize)
    y = numpy.log2(x) + numpy.cos(numpy.pi * x / 2)
    x = torch.Tensor(x).view(-1, 1)
    y = torch.Tensor(y).view(-1, 1)

    # 划分
    set = torch.randperm(Num_SampleSize)
    train_set = set[:int(Num_SampleSize*0.8)]
    val_set = set[int(Num_SampleSize*0.8):int(Num_SampleSize*0.9)]
    test_set = set[int(Num_SampleSize*0.9):]

    x_train, y_train = x[train_set], y[train_set]
    x_val, y_val = x[val_set], y[val_set]
    x_test, y_test = x[test_set], y[test_set]

    torch.save(x_train, f'src/data/data_x_train_N={Num_SampleSize}.pt')
    torch.save(y_train, f'src/data/data_y_train_N={Num_SampleSize}.pt')
    torch.save(x_val, f'src/data/data_x_val_N={Num_SampleSize}.pt')
    torch.save(y_val, f'src/data/data_y_val_N={Num_SampleSize}.pt')
    torch.save(x_test, f'src/data/data_x_test_N={Num_SampleSize}.pt')
    torch.save(y_test, f'src/data/data_y_test_N={Num_SampleSize}.pt')

def Data_Process(Num_SampleSize, setting_batch_size):
    x_train = torch.load(f'src/data/data_x_train_N={Num_SampleSize}.pt')
    y_train = torch.load(f'src/data/data_y_train_N={Num_SampleSize}.pt')
    x_val = torch.load(f'src/data/data_x_val_N={Num_SampleSize}.pt')
    y_val = torch.load(f'src/data/data_y_val_N={Num_SampleSize}.pt')
    x_test = torch.load(f'src/data/data_x_test_N={Num_SampleSize}.pt')
    y_test = torch.load(f'src/data/data_y_test_N={Num_SampleSize}.pt')
    print(x_train.dtype)

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
    data = "src/data"
    if not os.path.exists(data):
        os.makedirs(data)
    Generate_dataset(Num_SampleSize)