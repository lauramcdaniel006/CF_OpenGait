import copy
import os
# Set CuBLAS workspace config for deterministic behavior (required for CUDA >= 10.2)
# This MUST be set before importing torch or any CUDA operations
if 'CUBLAS_WORKSPACE_CONFIG' not in os.environ:
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import inspect
import logging
import torch
import numpy as np
import torch.nn as nn
import torch.autograd as autograd
import yaml
import random
from torch.nn.parallel import DistributedDataParallel as DDP
from collections import OrderedDict, namedtuple


class NoOp:
    def __getattr__(self, *args):
        def no_op(*args, **kwargs): pass
        return no_op


class Odict(OrderedDict):
    def append(self, odict):
        dst_keys = self.keys()
        for k, v in odict.items():
            if not is_list(v):
                v = [v]
            if k in dst_keys:
                if is_list(self[k]):
                    self[k] += v
                else:
                    self[k] = [self[k]] + v
            else:
                self[k] = v


def Ntuple(description, keys, values):
    if not is_list_or_tuple(keys):
        keys = [keys]
        values = [values]
    Tuple = namedtuple(description, keys)
    return Tuple._make(values)


def get_valid_args(obj, input_args, free_keys=[]):
    if inspect.isfunction(obj):
        expected_keys = inspect.getfullargspec(obj)[0]
    elif inspect.isclass(obj):
        expected_keys = inspect.getfullargspec(obj.__init__)[0]
    else:
        raise ValueError('Just support function and class object!')
    unexpect_keys = list()
    expected_args = {}
    for k, v in input_args.items():
        if k in expected_keys:
            expected_args[k] = v
        elif k in free_keys:
            pass
        else:
            unexpect_keys.append(k)
    if unexpect_keys != []:
        logging.info("Find Unexpected Args(%s) in the Configuration of - %s -" %
                     (', '.join(unexpect_keys), obj.__name__))
    return expected_args


def get_attr_from(sources, name):
    try:
        return getattr(sources[0], name)
    except:
        return get_attr_from(sources[1:], name) if len(sources) > 1 else getattr(sources[0], name)


def is_list_or_tuple(x):
    return isinstance(x, (list, tuple))


def is_bool(x):
    return isinstance(x, bool)


def is_str(x):
    return isinstance(x, str)


def is_list(x):
    return isinstance(x, list) or isinstance(x, nn.ModuleList)


def is_dict(x):
    return isinstance(x, dict) or isinstance(x, OrderedDict) or isinstance(x, Odict)


def is_tensor(x):
    return isinstance(x, torch.Tensor)


def is_array(x):
    return isinstance(x, np.ndarray)


def ts2np(x):
    return x.cpu().data.numpy()


def ts2var(x, **kwargs):
    return autograd.Variable(x, **kwargs).cuda()


def np2var(x, **kwargs):
    return ts2var(torch.from_numpy(x), **kwargs)


def list2var(x, **kwargs):
    return np2var(np.array(x), **kwargs)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def MergeCfgsDict(src, dst):
    for k, v in src.items():
        if (k not in dst.keys()) or (type(v) != type(dict())):
            dst[k] = v
        else:
            if is_dict(src[k]) and is_dict(dst[k]):
                MergeCfgsDict(src[k], dst[k])
            else:
                dst[k] = v


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def config_loader(path):
    with open(path, 'r') as stream:
        src_cfgs = yaml.safe_load(stream)
    
    if src_cfgs is None:
        raise ValueError(f"Config file '{path}' is empty or invalid. Please check the YAML syntax.")
    
    # Get absolute path to default.yaml regardless of current working directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # common.py is in opengait/utils/, so go up two levels to OpenGait root
    opengait_root = os.path.dirname(os.path.dirname(current_dir))
    default_config_path = os.path.join(opengait_root, "configs", "default.yaml")
    
    with open(default_config_path, 'r') as stream:
        dst_cfgs = yaml.safe_load(stream)
    
    if dst_cfgs is None:
        raise ValueError(f"Default config file '{default_config_path}' is empty or invalid.")
    
    MergeCfgsDict(src_cfgs, dst_cfgs)
    return dst_cfgs


def set_seed(s, device_id=None):
    """Set all random seeds (random, numpy, torch, torch.cuda).
    
    Keeps data sampling deterministic while allowing CUDA optimizations for better accuracy.
    
    Args:
        s: Seed value to set
        device_id: If provided, sets seed only for this specific CUDA device.
                   If None, sets seed for all CUDA devices (default behavior).
    """
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if device_id is not None:
        # Set seed for specific GPU device
        # In distributed training, each process only sees its own GPU (rank=device_id)
        # So we can set the device and then set the seed for that device
        if torch.cuda.is_available():
            torch.cuda.set_device(device_id)
            torch.cuda.manual_seed(s)  # Sets seed for current device only
    else:
        # Set seed for all CUDA devices (original behavior)
        torch.cuda.manual_seed_all(s)
    # CUDA determinism settings are controlled by init_seeds() function
    # Don't override here - let init_seeds() handle it based on cuda_deterministic parameter

def init_seeds(seed=0, cuda_deterministic=True, device_id=None):
    """Initialize seeds and set CUDA deterministic mode.
    
    For simple seed setting, use set_seed() instead.
    
    Args:
        seed: Seed value to set
        cuda_deterministic: Whether to use deterministic CUDA algorithms
        device_id: If provided, sets seed only for this specific CUDA device.
                   If None, sets seed for all CUDA devices (default behavior).
    """
    set_seed(seed, device_id=device_id)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Enable deterministic algorithms for full reproducibility
        # Note: This may cause errors for some operations - disable if needed
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def handler(signum, frame):
    logging.info('Ctrl+c/z pressed')
    os.system(
        "kill $(ps aux | grep main.py | grep -v grep | awk '{print $2}') ")
    logging.info('process group flush!')


def ddp_all_gather(features, dim=0, requires_grad=True):
    '''
        inputs: [n, ...]
    '''

    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    feature_list = [torch.ones_like(features) for _ in range(world_size)]
    torch.distributed.all_gather(feature_list, features.contiguous())

    if requires_grad:
        feature_list[rank] = features
    feature = torch.cat(feature_list, dim=dim)
    return feature


# https://github.com/pytorch/pytorch/issues/16885
class DDPPassthrough(DDP):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
    
    def _set_static_graph(self):
        """Passthrough for _set_static_graph to support gradient checkpointing with DDP"""
        # Call the parent DDP's _set_static_graph method directly
        return DDP._set_static_graph(self)


def get_ddp_module(module, find_unused_parameters=False, **kwargs):
    if len(list(module.parameters())) == 0:
        # for the case that loss module has not parameters.
        return module
    device = torch.cuda.current_device()
    
    # Check if model uses gradient checkpointing (common with SwinGait)
    use_checkpoint = getattr(module, 'use_checkpoint', False)
    
    # Create DDP module
    module = DDPPassthrough(module, device_ids=[device], output_device=device,
                            find_unused_parameters=find_unused_parameters, **kwargs)
    
    # If checkpointing is enabled, set static_graph to avoid DDP conflicts with reentrant backward passes
    if use_checkpoint:
        # Try setting static_graph - this tells DDP the computation graph doesn't change
        # which is needed for gradient checkpointing to work with DDP
        if hasattr(module, '_set_static_graph'):
            module._set_static_graph()
        elif hasattr(module, 'static_graph'):
            # For newer PyTorch versions, static_graph might be a property to set
            try:
                module.static_graph = True
            except:
                pass
    
    return module


def params_count(net):
    n_parameters = sum(p.numel() for p in net.parameters())
    return 'Parameters Count: {:.5f}M'.format(n_parameters / 1e6)
