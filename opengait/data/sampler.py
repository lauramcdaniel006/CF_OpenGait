import math
import random
import torch
import torch.distributed as dist
import torch.utils.data as tordata
import numpy as np
from utils import set_seed

# Global batch counter for deterministic seeding across batches
# Note: In distributed training, each rank has its own copy, so we synchronize via broadcast
# Start at -1 so first increment makes it 0, giving batch 0 seed 42
_batch_counter = -1

def get_batch_seed():
    """Get deterministic seed for current batch based on global counter.
    
    In distributed training, synchronizes counter across all ranks by broadcasting from rank 0.
    
    During evaluation (when counter is -1), returns GPU rank as base seed for consistency with main.py.
    """
    global _batch_counter
    
    # Use GPU rank as base seed (GPU 0 = 0, GPU 1 = 1) to match main.py
    if dist.is_initialized():
        base_seed = dist.get_rank()
    else:
        base_seed = 0  # Single GPU uses seed 0
    
    # Synchronize counter across ranks in distributed training
    if dist.is_initialized():
        # Get current device for this rank
        rank = dist.get_rank()
        device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
        
        # Broadcast counter from rank 0 to all ranks
        if rank == 0:
            counter_tensor = torch.tensor(_batch_counter, dtype=torch.long, device=device)
        else:
            counter_tensor = torch.tensor(0, dtype=torch.long, device=device)  # Placeholder
        
        torch.distributed.broadcast(counter_tensor, src=0)
        _batch_counter = int(counter_tensor.item())
    
    # During evaluation, counter is -1 (not incremented by InferenceSampler)
    # Return base_seed (GPU rank) for consistency with main.py
    if _batch_counter == -1:
        return base_seed
    
    return base_seed + _batch_counter

def increment_batch_counter():
    """Increment global batch counter. Called by sampler when yielding a batch.
    
    Only rank 0 increments, then broadcasts to all ranks.
    All ranks synchronize before incrementing to ensure perfect alignment.
    """
    global _batch_counter
    if dist.is_initialized():
        # Synchronize all ranks before incrementing to ensure perfect alignment
        torch.distributed.barrier()
        
        # Get current device for this rank
        rank = dist.get_rank()
        device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
        
        if rank == 0:
            _batch_counter += 1
            # Broadcast new counter value to all ranks
            counter_tensor = torch.tensor(_batch_counter, dtype=torch.long, device=device)
        else:
            # Other ranks receive the broadcast
            counter_tensor = torch.tensor(0, dtype=torch.long, device=device)  # Placeholder
        
        torch.distributed.broadcast(counter_tensor, src=0)
        _batch_counter = int(counter_tensor.item())
    else:
        # Single GPU case
        _batch_counter += 1

def reset_batch_counter():
    """Reset batch counter. Called at start of training.
    
    Resets on all ranks and synchronizes in distributed training.
    Starts at -1 so first increment makes it 0, giving batch 0 seed 42.
    """
    global _batch_counter
    
    # Synchronize reset across ranks
    if dist.is_initialized():
        # Get current device for this rank
        rank = dist.get_rank()
        device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
        
        # Rank 0 sets the value, then broadcasts
        if rank == 0:
            _batch_counter = -1
            counter_tensor = torch.tensor(-1, dtype=torch.long, device=device)
        else:
            counter_tensor = torch.tensor(0, dtype=torch.long, device=device)  # Placeholder
        
        torch.distributed.broadcast(counter_tensor, src=0)
        _batch_counter = int(counter_tensor.item())
    else:
        # Single GPU case
        _batch_counter = -1


class TripletSampler(tordata.sampler.Sampler):
    def __init__(self, dataset, batch_size, batch_shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        if len(self.batch_size) != 2:
            raise ValueError(
                "batch_size should be (P x K) not {}".format(batch_size))
        self.batch_shuffle = batch_shuffle

        self.world_size = dist.get_world_size()
        if (self.batch_size[0]*self.batch_size[1]) % self.world_size != 0:
            raise ValueError("World size ({}) is not divisible by batch_size ({} x {})".format(
                self.world_size, batch_size[0], batch_size[1]))
        self.rank = dist.get_rank()

    def __iter__(self):
        while True:
            # Increment counter BEFORE generating batch so CollateFn sees correct seed
            # This ensures sampler and CollateFn use the same seed for the same batch
            increment_batch_counter()
            
            # Seed random state deterministically for this batch
            batch_seed = get_batch_seed()
            set_seed(batch_seed)
            
            sample_indices = []
            pid_list = sync_random_sample_list(
                self.dataset.label_set, self.batch_size[0], batch_seed=batch_seed)

            for pid in pid_list:
                indices = self.dataset.indices_dict[pid]
                indices = sync_random_sample_list(
                    indices, k=self.batch_size[1], batch_seed=batch_seed)
                sample_indices += indices

            if self.batch_shuffle:
                sample_indices = sync_random_sample_list(
                    sample_indices, len(sample_indices), batch_seed=batch_seed)

            total_batch_size = self.batch_size[0] * self.batch_size[1]
            total_size = int(math.ceil(total_batch_size /
                                       self.world_size)) * self.world_size
            sample_indices += sample_indices[:(
                total_batch_size - len(sample_indices))]

            sample_indices = sample_indices[self.rank:total_size:self.world_size]
            
            yield sample_indices

    def __len__(self):
        return len(self.dataset)


def sync_random_sample_list(obj_list, k, common_choice=False, batch_seed=None):
    """
    Deterministically sample k items from obj_list.
    
    Args:
        obj_list: List to sample from
        k: Number of items to sample
        common_choice: Whether to use random.choices (with replacement)
        batch_seed: Seed for this batch (if None, uses current random state)
    """
    # Get current device for this rank
    rank = dist.get_rank()
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    
    # Initialize generator (will be set on rank 0, None on other ranks)
    generator = None
    
    # Only rank 0 generates random indices, then broadcasts to all ranks
    if rank == 0:
        if batch_seed is not None:
            # Use deterministic generator for torch.randperm
            # Generator must be on same device as the tensor operation
            generator = torch.Generator(device=device)
            generator.manual_seed(batch_seed)
        
        # Perform sampling on rank 0
        if common_choice:
            idx = random.choices(range(len(obj_list)), k=k)
            idx = torch.tensor(idx, dtype=torch.long, device=device)
        elif len(obj_list) < k:
            idx = random.choices(range(len(obj_list)), k=k)
            idx = torch.tensor(idx, dtype=torch.long, device=device)
        else:
            # Use generator for deterministic randperm
            if generator is not None:
                idx = torch.randperm(len(obj_list), generator=generator, device=device)[:k]
            else:
                idx = torch.randperm(len(obj_list), device=device)[:k]
    else:
        # Other ranks create placeholder tensor on CUDA
        idx = torch.zeros(k, dtype=torch.long, device=device)
    
    # Broadcast from rank 0 to all ranks
    torch.distributed.broadcast(idx, src=0)
    idx = idx.tolist()
    return [obj_list[i] for i in idx]


class InferenceSampler(tordata.sampler.Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

        self.size = len(dataset)
        indices = list(range(self.size))

        world_size = dist.get_world_size()
        rank = dist.get_rank()

        if batch_size % world_size != 0:
            raise ValueError("World size ({}) is not divisible by batch_size ({})".format(
                world_size, batch_size))

        if batch_size != 1:
            complement_size = math.ceil(self.size / batch_size) * \
                batch_size
            indices += indices[:(complement_size - self.size)]
            self.size = complement_size

        batch_size_per_rank = int(self.batch_size / world_size)
        indx_batch_per_rank = []

        for i in range(int(self.size / batch_size_per_rank)):
            indx_batch_per_rank.append(
                indices[i*batch_size_per_rank:(i+1)*batch_size_per_rank])

        self.idx_batch_this_rank = indx_batch_per_rank[rank::world_size]

    def __iter__(self):
        yield from self.idx_batch_this_rank

    def __len__(self):
        return len(self.dataset)


class CommonSampler(tordata.sampler.Sampler):
    def __init__(self,dataset,batch_size,batch_shuffle):

        self.dataset = dataset
        self.size = len(dataset)
        self.batch_size = batch_size
        if isinstance(self.batch_size,int)==False:
            raise ValueError(
                "batch_size shoude be (B) not {}".format(batch_size))
        self.batch_shuffle = batch_shuffle
        
        self.world_size = dist.get_world_size()
        if self.batch_size % self.world_size !=0:
            raise ValueError("World size ({}) is not divisble by batch_size ({})".format(
                self.world_size, batch_size))
        self.rank = dist.get_rank() 
    
    def __iter__(self):
        while True:
            # Increment counter BEFORE generating batch so CollateFn sees correct seed
            increment_batch_counter()
            
            # Seed random state deterministically for this batch
            batch_seed = get_batch_seed()
            set_seed(batch_seed)
            
            indices_list = list(range(self.size))
            sample_indices = sync_random_sample_list(
                    indices_list, self.batch_size, common_choice=True, batch_seed=batch_seed)
            total_batch_size =  self.batch_size
            total_size = int(math.ceil(total_batch_size /
                                       self.world_size)) * self.world_size
            sample_indices += sample_indices[:(
                total_batch_size - len(sample_indices))]
            sample_indices = sample_indices[self.rank:total_size:self.world_size]
            
            yield sample_indices

    def __len__(self):
        return len(self.dataset)

# **************** For GaitSSB ****************
# Fan, et al: Learning Gait Representation from Massive Unlabelled Walking Videos: A Benchmark, T-PAMI2023
class BilateralSampler(tordata.sampler.Sampler):
    def __init__(self, dataset, batch_size, batch_shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.batch_shuffle = batch_shuffle

        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        self.dataset_length = len(self.dataset)
        self.total_indices = list(range(self.dataset_length))
        self.count = 0

    def __iter__(self):
        batch_size = self.batch_size[0] * self.batch_size[1]
        while True:
            if self.count == 0 or (self.count * batch_size) % self.dataset_length == 0:
                shuffled_indices = self.total_indices.copy()
                random.shuffle(shuffled_indices)
                self.total_indices = shuffled_indices

            sampled_indices = self.total_indices[self.count*batch_size:(self.count+1)*batch_size]
            sampled_indices = sync_random_sample_list(sampled_indices, len(sampled_indices))

            total_size = int(math.ceil(batch_size / self.world_size)) * self.world_size
            sampled_indices += sampled_indices[:(batch_size - len(sampled_indices))]

            sampled_indices = sampled_indices[self.rank:total_size:self.world_size]
            self.count += 1

            yield sampled_indices * 2

    def __len__(self):
        return len(self.dataset)