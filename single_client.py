from torch.multiprocessing import Process

from src.models import *
from src.options import args_parser, init_process
from src.client_settings import Client

if __name__ == "__main__":
    args = args_parser()
    rank = 1
    processes = []
    client = Client(rank, args)
    for i in range(args.round):
        p = Process(target=init_process, args=(
            rank, args.world_size, client.main))  # ローカルモデルからグローバルモデルの生成
        p.start()
        processes.append(p)
        for p in processes:
            p.join()
