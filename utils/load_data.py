from utils.relational_table_preprocessor import image_preprocess_dl
import pandas as pd
import numpy as np
from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader,TensorDataset,ConcatDataset
from collections import Counter
import os
from sklearn.datasets import fetch_openml
import random


def load_mnist_image(args):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset_train = datasets.MNIST('./data/mnist', train=True, download=True, transform=transform) 
    data_train =torch.unsqueeze(dataset_train.data, dim=1).to('cpu').detach().numpy().copy()

    labels_train = dataset_train.targets

    dataset_test = datasets.MNIST('./data/mnist', train=False, download=True, transform=transform)
    data_test = torch.unsqueeze(dataset_test.data, dim=1).to('cpu').detach().numpy().copy()
    
    labels_test = dataset_test.targets
    label_list = torch.cat((labels_train,labels_test), dim=0)
    


    data = np.block([[[[data_train]]], [[[data_test]]]])
    
    args.class_num = len(np.unique(label_list))
    [_, _, _, _, _, train_data_local_dict, test_data_local_dict, args.class_num] = image_preprocess_dl(args,
                                                                                                     data,
                                                                                                     label_list,
                                                                                                     test_partition=1/7)
    for key in train_data_local_dict.keys():
        torch.save(train_data_local_dict[key], os.path.join(args.datapath,f"data_worker{key+1}_train.pt"))
        print(dict(sorted(dict(Counter(train_data_local_dict[key].dataset[:][1].numpy().tolist())).items())))
        torch.save(test_data_local_dict[key], os.path.join(args.datapath,f"data_worker{key+1}_test.pt"))

def load_cifar10_image(args):
    trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset_train = datasets.CIFAR10('./data/cifar10', train=True, download=True, transform=trans_cifar)
    data_train = dataset_train.data

    labels_train = dataset_train.targets

    dataset_test = datasets.CIFAR10('./data/cifar10', train=False, download=True, transform=trans_cifar)
    data_test = dataset_test.data
    
    labels_test = dataset_test.targets


    data = np.block([[[[data_train]]], [[[data_test]]]])
    data = np.transpose(data, (0, 3, 1, 2))

    label_list = labels_train+labels_test
    args.class_num = len(np.unique(label_list))

    [_, _, _, _, _, train_data_local_dict, test_data_local_dict, args.class_num] = image_preprocess_dl(args,
                                                                                                        data,
                                                                                                        label_list,
                                                                                                        test_partition=0.2)
    for key in train_data_local_dict.keys():

        torch.save(train_data_local_dict[key], os.path.join(args.datapath,f"data_worker{key+1}_train.pt"))
        print(dict(sorted(dict(Counter(train_data_local_dict[key].dataset[:][1].numpy().tolist())).items())))
        torch.save(test_data_local_dict[key], os.path.join(args.datapath,f"data_worker{key+1}_test.pt"))

def load_mnist_image_iid(args):
    val = 10#データセットを何個に分けるか
    transform = transforms.Compose([transforms.ToTensor()])
    dataset_train = datasets.MNIST('./data/mnist', train=True, download=True, transform=transform) 

    dataset_test = datasets.MNIST('./data/mnist', train=False, download=True, transform=transform)
    
    train_data_local_dict = {}
    test_data_local_dict = {}
    dataset_train_list = []
    dataset_test_list = []
    train_size = [int(len(dataset_train) / val)]*val
    test_size = [int(len(dataset_test) / val)]*val

    print('train dataset size ={}'.format(train_size[0]))


    for i in range(args.client_num_in_total):
        cnt = i % val
        if cnt == 0:
            dataset_train_list = torch.utils.data.random_split(dataset_train, train_size)
            dataset_test_list = torch.utils.data.random_split(dataset_test, test_size)


        train_loader = DataLoader(dataset_train_list[cnt], 
                        batch_size=args.batch_size,
                        shuffle=True)

        test_loader = DataLoader(dataset_test_list[cnt],
                        batch_size=args.batch_size,
                        shuffle=True)

        train_data_local_dict[i]=train_loader
        test_data_local_dict[i]=test_loader

     
    args.class_num = 10
    
    for key in train_data_local_dict.keys():
        torch.save(train_data_local_dict[key], os.path.join(args.datapath,f"data_worker{key+1}_train.pt"))
        torch.save(test_data_local_dict[key], os.path.join(args.datapath,f"data_worker{key+1}_test.pt"))


def load_cifar10_image_iid(args):
    val = 10
    trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset_train = datasets.CIFAR10('./data/cifar10', train=True, download=True, transform=trans_cifar)

    dataset_test = datasets.CIFAR10('./data/cifar10', train=False, download=True, transform=trans_cifar)

    train_data_local_dict = {}
    test_data_local_dict = {}
    dataset_train_list = []
    dataset_test_list = []
    train_val_size = [int(len(dataset_train) / val)]*val
    test_val_size = [int(len(dataset_test) / val)]*val


    for i in range(args.client_num_in_total):
        cnt = i % val
        if cnt == 0:
            dataset_train_list = torch.utils.data.random_split(dataset_train, train_val_size)
            dataset_test_list = torch.utils.data.random_split(dataset_test, test_val_size)


        train_loader = DataLoader(dataset_train_list[cnt], 
                        batch_size=args.batch_size,
                        shuffle=True)

        test_loader = DataLoader(dataset_test_list[cnt],
                        batch_size=args.batch_size,
                        shuffle=True)

        train_data_local_dict[i]=train_loader
        test_data_local_dict[i]=test_loader

     
    args.class_num = 10

    for key in train_data_local_dict.keys():
        torch.save(train_data_local_dict[key], os.path.join(args.datapath,f"data_worker{key+1}_train.pt"))
        torch.save(test_data_local_dict[key], os.path.join(args.datapath,f"data_worker{key+1}_test.pt"))

if __name__ == '__main__':
    pass