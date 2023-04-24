import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process


def run_gather(rank, size):
    """ Copies tensor from all processes to tensor_list, on all processes. """
    group = dist.new_group([0, 1, 2])
    tensor = torch.tensor([rank])
    tensor_list = [torch.tensor([0]) for _ in range(dist.get_world_size())]

    dist.gather(tensor, tensor_list, group=group, dst=0)
    print('Rank ', rank, ' has data ', tensor_list)


def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 3
    rank = 0
    processes = []
    p = Process(target=init_process, args=(
        rank, size, run_gather))
    p.start()
    processes.append(p)

    for p in processes:
        p.join()
