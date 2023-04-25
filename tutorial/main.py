import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process


def run_all_reduce(rank, size):
    """  Applies op to every tensor and the result is stored in all processes. """
    group = dist.new_group([0, 1, 2])
    tensor = torch.tensor([rank])
    # dist.reduce_op.PRODUCT # dist.reduce_op.MAX # dist.reduce_op.MIN
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    print('Rank ', rank, ' has data ', tensor[0])


def run_all_gather(rank, size):
    """ Copies tensor from all processes to tensor_list, on all processes. """
    group = dist.new_group([0, 1, 2])
    tensor = torch.tensor([rank])
    tensor_list = [torch.tensor([0]) for _ in range(dist.get_world_size())]

    dist.all_gather(tensor_list, tensor, group=group)
    print('Rank ', rank, ' has data ', tensor_list)


def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '10.24.30.12'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 3
    processes = []
    for rank in range(size):
        p = Process(target=init_process, args=(
            rank, size, run_gather))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
