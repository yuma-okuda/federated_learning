import os 
import re
import torch
import torch.multiprocessing as mp
import torch.distributed.rpc as rpc

from utils.load_data import load_mnist_image, load_cifar10_image, load_mnist_image_iid,load_cifar10_image_iid
from src.options import *
if __name__ == "__main__":
    args = args_parser()
    if args.dataset == "mnist":
        print("mnist")
        if args.iid == True:
            print("make iid data")
            load_mnist_image_iid(args)
        else:
            print("make non-iid data")
            load_mnist_image(args)#client数に応じて、data_workerN_{train,test}.ptにデータを保存する

        

    elif args.dataset == "cifar10":
        print("cifar10")
        if args.iid == True:
            print("make iid data")
            load_cifar10_image_iid(args)
        else:
            print("make non-iid data")
            load_cifar10_image(args)