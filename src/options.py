#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse
import os
import torch.distributed as dist
import slackweb


def args_parser():
    parser = argparse.ArgumentParser(
        description='Federated Learning Initialization')
    parser.add_argument('--world_size', type=int, default=3,
                        help='The world size which is equal to 1 server + (world size - 1) clients')
    parser.add_argument('--epochs', type=int, default=5,
                        help='The number of epochs to run on the client training each iteration')
    parser.add_argument('--round', type=int, default=5,
                        help='The number of rounds to communication between clients and server')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='The batch size during the epoch training')
    parser.add_argument('--partition_alpha', type=float, default=0.5,
                        help='Number to describe the uniformity during sampling (heterogenous data generation for LDA)')
    parser.add_argument('--datapath', type=str, default="./data",
                        help='folder path to all the local datasets')
    parser.add_argument('--dataset', type=str, default="mnist",
                        help='The dataset for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate of local client (Adam)')
    parser.add_argument('--model', type=int, default=1, help='Select model')
    parser.add_argument('--iid', type=str, default=True, help='iid or non-iid')
    args = parser.parse_args()

    args.client_num_in_total = args.world_size - 1
    args.datapath = args.datapath + "/" + args.dataset

    return args


def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = 'localhost'  # yarrow = '10.24.30.12'
    os.environ['MASTER_PORT'] = '9000'
    # os.environ['GLOO_SOCKET_IFNAME'] = 'enp4s0'

    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)
