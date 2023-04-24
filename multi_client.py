from torch.multiprocessing import Process
import torch.multiprocessing as mp

from src.models import *
from src.options import args_parser, init_process
from src.client_settings import Client


def exec_client(rank, world_size, args):
    if rank == 0:
        pass
    else:
        processes = []
        client = Client(rank, args)
        for i in range(args.round):
            args.i = i
            p = Process(target=init_process, args=(
                rank, args.world_size, client.main))
            p.start()
            processes.append(p)
            for p in processes:
                p.join()


if __name__ == "__main__":
    args = args_parser()

    mp.spawn(exec_client,
             args=(args.world_size, args),
             nprocs=args.world_size,
             join=True
             )
