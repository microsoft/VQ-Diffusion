"""
This file contains primitives for multi-gpu communication.
This is useful when doing distributed training.
"""

import pickle

import torch
import torch.distributed as dist
# from diffdist.functional import all_gather as better_all_gather


class Comm(object):
    def __init__(self, local_rank=0):
        self.local_rank = 0

    @property
    def world_size(self):
        if not dist.is_available():
            return 1
        if not dist.is_initialized():
            return 1
        return dist.get_world_size()

    @property
    def rank(self):
        if not dist.is_available():
            return 0
        if not dist.is_initialized():
            return 0
        return dist.get_rank()

    @property
    def local_rank(self):
        if not dist.is_available():
            print("****************** yes1")
            return 0
        if not dist.is_initialized():
            print("****************** yes2")
            return 0
        print("****************** yes3", self._local_rank)
        return self._local_rank

    @local_rank.setter
    def local_rank(self, value):
        if not dist.is_available():
            self._local_rank = 0
        if not dist.is_initialized():
            self._local_rank = 0
        self._local_rank = value

    @property
    def head(self):
        return 'Rank[{}/{}]'.format(self.rank, self.world_size)
   
    def is_main_process(self):
        return self.rank == 0

    def synchronize(self):
        """
        Helper function to synchronize (barrier) among all processes when
        using distributed training
        """
        if self.world_size == 1:
            return
        dist.barrier()


comm = Comm()


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = comm.world_size
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.LongTensor([tensor.numel()]).to("cuda")
    size_list = [torch.LongTensor([0]).to("cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,)).to("cuda"))
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,)).to("cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = comm.world_size
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.reduce(values, dst=0)
        if dist.get_rank() == 0 and average:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def gather_tensors(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor)
        for _ in range(comm.world_size)
    ]

    dist.all_gather(tensors_gather, tensor, async_op=False)
    # need to do this to restore propagation of the gradients
    tensors_gather[comm.rank] = tensor
    output = torch.cat(tensors_gather, dim=0)
    return output

def gather_tensors_fake(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor)
        for _ in range(comm.world_size)
    ]

    dist.all_gather(tensors_gather, tensor, async_op=False)
    # need to do this to restore propagation of the gradients
    tensors_gather[comm.rank] = tensor
    output = torch.cat(tensors_gather, dim=0)
    output = torch.cat([output,output.detach()],0)
    return output

def gather_nearby_tensors(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    step=comm.rank//2
    if comm.rank%2==0:
        nearby_rank=step*2+1
    else:
        nearby_rank=step*2
    cpu_tensor=tensor
    tensors_gather = [
        torch.ones_like(cpu_tensor)
        for _ in range(comm.world_size)
    ]
    dist.all_gather(tensors_gather, cpu_tensor, async_op=False)
    # need to do this to restore propagation of the gradients
    tensors_gather=[tensors_gather[nearby_rank].to(tensor.device),tensor]
    output = torch.cat(tensors_gather, dim=0)
    return output


def gather_tensors_with_gradient(x):
    """ collect all tensor from all GPUs
    args:
        x: shape (mini_batch, ...)
    returns:
        shape (mini_batch * num_gpu, ...)
    """
    x = x.contiguous()
    out_list = [torch.zeros_like(x, device=x.device, dtype=x.dtype) for _ in range(comm.world_size)]
    out_list = better_all_gather(out_list, x)
    return torch.cat(out_list, dim=0)

gather_funcs={
    "ALL":gather_tensors,
    "NEAR":gather_nearby_tensors,
    "GRAD":gather_tensors_with_gradient,
    "FAKE":gather_tensors_fake
}

from contextlib import contextmanager

@contextmanager
def torch_distributed_zero_first():
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    local_rank=comm.local_rank
    if local_rank not in [-1, 0]:
        dist.barrier(device_ids=[local_rank])
    yield
    if local_rank == 0:
        dist.barrier(device_ids=[0])
