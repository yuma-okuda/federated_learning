import time
import torch
import torch.nn as nn
import torchvision
import torch.distributed as dist
import torchvision.transforms as transforms
from torch.distributed.rpc import RRef
import torch.distributed.rpc as rpc
import torch.distributed.autograd as dist_autograd
from torch.distributed.optim import DistributedOptimizer
import os
from collections import Counter
from copy import deepcopy
import matplotlib.pyplot as plt

import copy

from src.models import *


class Server(object):
    def __init__(self, args):
        self.dev = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.args = args
        self.output_dict = {}
        if args.dataset == 'mnist':
            if args.model == 1:
                self.model_s = model_mnist()
            elif args.model == 2:
                self.model_s = model_mnist_resnet18()
            elif args.model == 3:
                self.model_s = model_mnist_resnet50()

        elif args.dataset == 'cifar10':
            self.model_s = model_cifar10()

    def main(self, rank, size):
        """ Simple point-to-point communication. """
        group = dist.new_group([i for i in range(size)])
        self.output = [self.model_s for _ in range(dist.get_world_size())]
        dist.gather_object(self.model_s, object_gather_list=self.output,
                           dst=0, group=group)  # ローカルモデルの受信
        for i in range(len(self.output)):
            self.output[i] = self.output[i].state_dict()  # それぞれのモデルを配列に代入
        w_glob = self.avg(self.output[1:])  # グローバルモデルの生成
        self.model_s.load_state_dict(w_glob)
        dist.broadcast_object_list(
            object_list=[self.model_s], src=0, group=group)  # グローバルモデルの送信

    def avg(self, w):  # 平均化処理
        w_avg = copy.deepcopy(w[0])
        for k in w_avg.keys():  # それぞれの重みやバイアスを代入
            for i in range(len(w)):
                if i == 0:
                    w_avg[k] = w[i][k]
                else:
                    w_avg[k] += w[i][k]
            w_avg[k] = torch.div(w_avg[k], len(w))  # それぞれの要素を総数で割って平均化
        return w_avg
