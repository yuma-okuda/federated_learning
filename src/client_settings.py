#!/usr/bin/env python
import os
import time
import torch
import torch.distributed as dist
from torch.multiprocessing import Process

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.models import *
from src.options import args_parser
import sys


class Client(object):
    def __init__(self, rank, args):
        self.dev = torch.device(f"cuda:0")
        self.rank = rank
        self.epochs = args.epochs
        self.args = args
        # self.start_logger(args)
        if args.dataset == 'mnist':
            if args.model == 1:
                self.model_c = model_mnist()
            elif args.model == 2:
                self.model_c = model_mnist_resnet18()
            elif args.model == 3:
                self.model_c = model_mnist_resnet50()
        elif args.dataset == 'cifar10':
            self.model_c = model_cifar10()

        self.optimizer = torch.optim.Adam(
            self.model_c.parameters(), lr=args.lr)

        self.criterion = nn.CrossEntropyLoss()

        self.load_data(args)

    def train(self, args):
        self.model_c.to(self.dev)
        if args.i != 0:
            self.model_c.load_state_dict(
                torch.load(f'./data/model_rank{self.rank}.pth'))

        for epoch in range(args.epochs):
            self.model_c.train()
            for i, data in enumerate(self.train_dataloader):
                inputs, labels = data
                inputs = inputs.to(self.dev)
                labels = labels.to(self.dev)
                self.optimizer.zero_grad()

                output = self.model_c(inputs)
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()
                if (i+1) % 10 == 0:
                    print("[rank: %d][Epoch: %d/%d] [Batch: %d/%d] [Loss: %f]"
                          % (self.rank, epoch+1, self.epochs, i+1, len(self.train_dataloader), loss)
                          )
        self.model_c.to('cpu')

    def test(self, args):
        self.model_c.to(self.dev)
        self.model_c.eval()

        total = 0
        correct = 0
        pred_list = []
        target_list = []
        with torch.no_grad():
            for i, data in enumerate(self.test_dataloader):
                inputs, labels = data
                inputs = inputs.to(self.dev)
                labels = labels.to(self.dev)
                test_output = self.model_c(inputs)
                total += labels.size(0)
                predicted = torch.max(test_output.data, 1)

                for i in range(len(predicted[1])):
                    if (predicted[1][i] == labels[i]):
                        correct = correct + 1
                    target_list.append(labels[i].to(
                        'cpu').detach().numpy().copy())

                    pred_list.append(predicted[1][i].to(
                        'cpu').detach().numpy().copy())
        acc = correct / total
        print(f'rank: {self.rank} acc: {acc}')

    def load_data(self, args):  # データのロード
        self.train_dataloader = torch.load(os.path.join(
            args.datapath, f"data_worker{self.rank}_train.pt"))
        self.test_dataloader = torch.load(os.path.join(
            args.datapath, f"data_worker{self.rank}_test.pt"))

    def main(self, rank, size):

        group = dist.new_group([i for i in range(size)])  # グループ作成

        self.train(self.args)

        dist.gather_object(self.model_c, dst=0, group=group)  # クライアントモデルの送信
        avg_model_list = [None]
        dist.broadcast_object_list(
            object_list=avg_model_list, src=0, group=group)  # グローバルモデルの受け取り
        torch.save(avg_model_list[0].state_dict(),
                   f'./data/model_rank{rank}.pth')

        self.test(self.args)


if __name__ == '__main__':
    pass
