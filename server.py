#!/usr/bin/env python
import torch.multiprocessing as mp
from torch.multiprocessing import Process
from src.models import *
from src.options import args_parser, init_process
from src.server_settings import Server


def exec_server(rank, args):
    processes = []
    server = Server(args)
    for i in range(args.round):
        print(f"Iteration {i+1} Start")
        args.i = i
        p = Process(target=init_process, args=(
            rank, args.world_size, server.main))
        p.start()
        processes.append(p)
        for p in processes:
            p.join()  # ここでmain関数が動いている


if __name__ == "__main__":
    args = args_parser()
    rank = 0
    exec_server(rank=rank, args=args)
