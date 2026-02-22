"""The base model definition.

This module defines the abstract meta model class and base model class. In the base model,
 we define the basic model functions, like get_loader, build_network, and run_train, etc.
 The api of the base model is run_train and run_test, they are used in `opengait/main.py`.

Typical usage:

BaseModel.run_train(model)
BaseModel.run_test(model)
"""
import torch
import numpy as np
import os.path as osp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tordata

from tqdm import tqdm
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from abc import ABCMeta
from abc import abstractmethod

from . import backbones
from .loss_aggregator import LossAggregator
from data.transform import get_transform
from data.collate_fn import CollateFn
from data.dataset import DataSet
import data.sampler as Samplers
from utils import Odict, mkdir, ddp_all_gather
from utils import get_valid_args, is_list, is_dict, np2var, ts2np, list2var, get_attr_from
from evaluation import evaluator as eval_functions
from utils import NoOp
from utils import get_msg_mgr

__all__ = ['BaseModel']


class MetaModel(metaclass=ABCMeta):
    """The necessary functions for the base model.

    This class defines the necessary functions for the base model, in the base model, we have implemented them.
    """
    @abstractmethod
    def get_loader(self, data_cfg):
        """Based on the given data_cfg, we get the data loader."""
        raise NotImplementedError

    @abstractmethod
    def build_network(self, model_cfg):
        """Build your network here."""
        raise NotImplementedError

    @abstractmethod
    def init_parameters(self):
        """Initialize the parameters of your network."""
        raise NotImplementedError

    @abstractmethod
    def get_optimizer(self, optimizer_cfg):
        """Based on the given optimizer_cfg, we get the optimizer."""
        raise NotImplementedError

    @abstractmethod
    def get_scheduler(self, scheduler_cfg):
        """Based on the given scheduler_cfg, we get the scheduler."""
        raise NotImplementedError

    @abstractmethod
    def save_ckpt(self, iteration):
        """Save the checkpoint, including model parameter, optimizer and scheduler."""
        raise NotImplementedError

    @abstractmethod
    def resume_ckpt(self, restore_hint):
        """Resume the model from the checkpoint, including model parameter, optimizer and scheduler."""
        raise NotImplementedError

    @abstractmethod
    def inputs_pretreament(self, inputs):
        """Transform the input data based on transform setting."""
        raise NotImplementedError

    @abstractmethod
    def train_step(self, loss_num) -> bool:
        """Do one training step."""
        raise NotImplementedError

    @abstractmethod
    def inference(self):
        """Do inference (calculate features.)."""
        raise NotImplementedError

    @abstractmethod
    def run_train(model):
        """Run a whole train schedule."""
        raise NotImplementedError

    @abstractmethod
    def run_test(model):
        """Run a whole test schedule."""
        raise NotImplementedError


class BaseModel(MetaModel, nn.Module):
    """Base model.

    This class inherites the MetaModel class, and implements the basic model functions, like get_loader, build_network, etc.

    Attributes:
        msg_mgr: the massage manager.
        cfgs: the configs.
        iteration: the current iteration of the model.
        engine_cfg: the configs of the engine(train or test).
        save_path: the path to save the checkpoints.

    """

    def __init__(self, cfgs, training):
        """Initialize the base model.

        Complete the model initialization, including the data loader, the network, the optimizer, the scheduler, the loss.

        Args:
        cfgs:
            All of the configs.
        training:
            Whether the model is in training mode.
        """

        super(BaseModel, self).__init__()
        self.msg_mgr = get_msg_mgr()
        self.cfgs = cfgs
        self.iteration = 0
        self.engine_cfg = cfgs['trainer_cfg'] if training else cfgs['evaluator_cfg']
        if self.engine_cfg is None:
            raise Exception("Initialize a model without -Engine-Cfgs-")

        if training and self.engine_cfg['enable_float16']:
            self.Scaler = GradScaler()
        self.save_path = osp.join('output/', cfgs['data_cfg']['dataset_name'],
                                  cfgs['model_cfg']['model'], self.engine_cfg['save_name'])

        self.build_network(cfgs['model_cfg'])
        self.init_parameters()
        self.log_trainable_frozen_params()
        self.trainer_trfs = get_transform(cfgs['trainer_cfg']['transform'])

        self.msg_mgr.log_info(cfgs['data_cfg'])
        if training:
            self.train_loader = self.get_loader(
                cfgs['data_cfg'], train=True)
        if not training or self.engine_cfg['with_test']:
            self.test_loader = self.get_loader(
                cfgs['data_cfg'], train=False)
            self.evaluator_trfs = get_transform(
                cfgs['evaluator_cfg']['transform'])
            self.val_loader = None

        self.device = torch.distributed.get_rank()
        torch.cuda.set_device(self.device)
        self.to(device=torch.device(
            "cuda", self.device))

        if training:
            try:
                from data.sampler import reset_batch_counter
                reset_batch_counter()
            except ImportError:
                pass  # Fallback if import fails
            
            self.loss_aggregator = LossAggregator(cfgs['loss_cfg'])
            self.optimizer = self.get_optimizer(self.cfgs['optimizer_cfg'])
            self.scheduler = self.get_scheduler(cfgs['scheduler_cfg'])
        self.train(training)
        restore_hint = self.engine_cfg['restore_hint']
        if restore_hint != 0:
            self.resume_ckpt(restore_hint)

    def log_trainable_frozen_params(self):
        """Log summary of trainable vs frozen parameters for any model."""
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        frozen_params = total_params - trainable_params
        
        # Define order for top-level components (main modules in architectural order)
        top_level_order = {
            'layer0': 0,
            'layer1': 1,
            'layer2': 2,
            'ulayer': 3,
            'transformer': 4,
            'FCs': 5,
            'BNNecks': 6,
            'TP': 7,
            'HPP': 8,
        }
        
        # Collect all modules with their parameter counts in model order
        layer_info = []
        for name, module in self.named_modules():
            # Skip if it's the root module itself
            if name == '':
                continue
            
            module_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            module_total = sum(p.numel() for p in module.parameters())
            
            # Only include modules that have parameters (skip empty modules)
            if module_total > 0:
                # Calculate depth for ordering
                depth = name.count('.')
                # Get top-level component name
                top_level = name.split('.')[0]
                # Get sort priority (lower number = appears first)
                priority = top_level_order.get(top_level, 999)  # Unknown components go last
                layer_info.append((priority, depth, name, module_trainable, module_total))
        
        self.msg_mgr.log_info("=" * 70)
        self.msg_mgr.log_info("LAYER FREEZING STATUS - INDIVIDUAL LAYERS")
        self.msg_mgr.log_info("=" * 70)
        self.msg_mgr.log_info("🔥 = Trainable (With gradients) | ❄️ = Frozen (No gradients)")
        self.msg_mgr.log_info("")
        
        # Sort by priority (top-level component order), then depth, then name
        layer_info.sort(key=lambda x: (x[0], x[1], x[2]))
        
        for priority, depth, name, module_trainable, module_total in layer_info:
            if module_trainable == 0:
                # Completely frozen
                emoji = "❄️"
                status = f"{module_total:>12,} params (FROZEN)"
            elif module_trainable == module_total:
                # Completely trainable
                emoji = "🔥"
                status = f"{module_trainable:>12,} params (TRAINABLE)"
            else:
                # Partially trainable
                emoji = "🔥❄️"
                status = f"{module_trainable:>12,} / {module_total:>12,} params (PARTIAL)"
            
            self.msg_mgr.log_info(f"{emoji} {name:55s} {status}")
        
        self.msg_mgr.log_info("")
        self.msg_mgr.log_info("=" * 70)
        self.msg_mgr.log_info("SUMMARY:")
        self.msg_mgr.log_info(f"  Total parameters:        {total_params:>12,}")
        self.msg_mgr.log_info(f"  🔥 Trainable parameters:    {trainable_params:>12,} ({trainable_params/total_params*100:.2f}%)")
        self.msg_mgr.log_info(f"  ❄️  Frozen parameters:       {frozen_params:>12,} ({frozen_params/total_params*100:.2f}%)")
        self.msg_mgr.log_info("=" * 70)

    def verify_gradient_status(self, check_frozen=True, check_trainable=True):
        """Verify that frozen parameters have no gradients and trainable ones do.
        
        This should be called after loss.backward() to verify freezing is working correctly.
        
        Args:
            check_frozen: If True, verify frozen parameters have no gradients
            check_trainable: If True, verify trainable parameters have gradients
        """
        frozen_params_with_grad = []
        trainable_params_without_grad = []
        frozen_params_count = 0
        trainable_params_count = 0
        
        for name, param in self.named_parameters():
            has_grad = param.grad is not None
            is_trainable = param.requires_grad
            
            if not is_trainable:
                frozen_params_count += param.numel()
                if has_grad and check_frozen:
                    frozen_params_with_grad.append((name, param.numel()))
            else:
                trainable_params_count += param.numel()
                if not has_grad and check_trainable:
                    trainable_params_without_grad.append((name, param.numel()))
        
        # Log verification results
        self.msg_mgr.log_info("=" * 60)
        self.msg_mgr.log_info("GRADIENT VERIFICATION (after backward pass)")
        self.msg_mgr.log_info("=" * 60)
        
        if check_frozen:
            if frozen_params_with_grad:
                self.msg_mgr.log_warning(f"⚠️  FROZEN PARAMETERS WITH GRADIENTS (SHOULD NOT HAPPEN): {len(frozen_params_with_grad)} parameters")
                for name, numel in frozen_params_with_grad[:10]:  # Show first 10
                    self.msg_mgr.log_warning(f"  • {name} ({numel:,} params)")
                if len(frozen_params_with_grad) > 10:
                    self.msg_mgr.log_warning(f"  ... and {len(frozen_params_with_grad) - 10} more")
            else:
                self.msg_mgr.log_info(f"✅ FROZEN PARAMETERS: {frozen_params_count:,} parameters have NO gradients (correct)")
        
        if check_trainable:
            if trainable_params_without_grad:
                self.msg_mgr.log_warning(f"⚠️  TRAINABLE PARAMETERS WITHOUT GRADIENTS: {len(trainable_params_without_grad)} parameters")
                for name, numel in trainable_params_without_grad[:10]:  # Show first 10
                    self.msg_mgr.log_warning(f"  • {name} ({numel:,} params)")
                if len(trainable_params_without_grad) > 10:
                    self.msg_mgr.log_warning(f"  ... and {len(trainable_params_without_grad) - 10} more")
            else:
                self.msg_mgr.log_info(f"✅ TRAINABLE PARAMETERS: {trainable_params_count:,} parameters have gradients (correct)")
        
        self.msg_mgr.log_info("=" * 60)
        
        # Return True if verification passed
        return len(frozen_params_with_grad) == 0 and len(trainable_params_without_grad) == 0

    def get_backbone(self, backbone_cfg):
        """Get the backbone of the model."""
        if is_dict(backbone_cfg):
            Backbone = get_attr_from([backbones], backbone_cfg['type'])
            valid_args = get_valid_args(Backbone, backbone_cfg, ['type'])
            return Backbone(**valid_args)
        if is_list(backbone_cfg):
            Backbone = nn.ModuleList([self.get_backbone(cfg)
                                      for cfg in backbone_cfg])
            return Backbone
        raise ValueError(
            "Error type for -Backbone-Cfg-, supported: (A list of) dict.")

    def build_network(self, model_cfg):
        if 'backbone_cfg' in model_cfg.keys():
            self.Backbone = self.get_backbone(model_cfg['backbone_cfg'])

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                if m.affine:
                    nn.init.normal_(m.weight.data, 1.0, 0.02)
                    nn.init.constant_(m.bias.data, 0.0)

    def get_loader(self, data_cfg, train=True, use_val_set=False):
        sampler_cfg = self.cfgs['trainer_cfg']['sampler'] if train else self.cfgs['evaluator_cfg']['sampler']
        dataset = DataSet(data_cfg, train, use_val_set=use_val_set)
        
        # Reset random seed for deterministic sampling
        from utils import set_seed
        seed = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        set_seed(seed)

        Sampler = get_attr_from([Samplers], sampler_cfg['type'])
        vaild_args = get_valid_args(Sampler, sampler_cfg, free_keys=[
            'sample_type', 'type'])
        sampler = Sampler(dataset, **vaild_args)
        
        loader = tordata.DataLoader(
            dataset=dataset,
            batch_sampler=sampler,
            collate_fn=CollateFn(dataset.label_set, sampler_cfg),
            num_workers=data_cfg['num_workers'])
        return loader

    def get_optimizer(self, optimizer_cfg):
        self.msg_mgr.log_info(optimizer_cfg)
        optimizer = get_attr_from([optim], optimizer_cfg['solver'])
        valid_arg = get_valid_args(optimizer, optimizer_cfg, ['solver'])
        optimizer = optimizer(
            filter(lambda p: p.requires_grad, self.parameters()), **valid_arg)
        return optimizer

    def get_scheduler(self, scheduler_cfg):
        self.msg_mgr.log_info(scheduler_cfg)
        Scheduler = get_attr_from(
            [optim.lr_scheduler], scheduler_cfg['scheduler'])
        valid_arg = get_valid_args(Scheduler, scheduler_cfg, ['scheduler'])
        scheduler = Scheduler(self.optimizer, **valid_arg)
        return scheduler

    def save_ckpt(self, iteration):
        if torch.distributed.get_rank() == 0:
            mkdir(osp.join(self.save_path, "checkpoints/"))
            save_name = self.engine_cfg['save_name']
            checkpoint = {
                'model': self.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'iteration': iteration}
            torch.save(checkpoint,
                       osp.join(self.save_path, 'checkpoints/{}-{:0>5}.pt'.format(save_name, iteration)))

    def _load_ckpt(self, save_name):
        load_ckpt_strict = self.engine_cfg['restore_ckpt_strict']

        checkpoint = torch.load(save_name, map_location=torch.device(
            "cuda", self.device))
        model_state_dict = checkpoint['model']

        if not load_ckpt_strict:
            current_state_dict = self.state_dict()
            filtered_state_dict = {}
            skipped_params = []
            
            for key, value in model_state_dict.items():
                if key in current_state_dict:
                    if current_state_dict[key].shape == value.shape:
                        filtered_state_dict[key] = value
                    else:
                        skipped_params.append(f"{key}: checkpoint shape {value.shape} vs model shape {current_state_dict[key].shape}")
                else:
                    skipped_params.append(f"{key}: not in current model")
            
            if skipped_params:
                self.msg_mgr.log_warning("-------- Skipped Parameters (size mismatch or missing) --------")
                for param in skipped_params:
                    self.msg_mgr.log_warning(f"  {param}")
            
            self.msg_mgr.log_info("-------- Restored Params List --------")
            self.msg_mgr.log_info(sorted(filtered_state_dict.keys()))
            
            model_state_dict = filtered_state_dict

        self.load_state_dict(model_state_dict, strict=load_ckpt_strict)
        if self.training:
            if not self.engine_cfg["optimizer_reset"] and 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                self.msg_mgr.log_warning(
                    "Restore NO Optimizer from %s !!!" % save_name)
            if not self.engine_cfg["scheduler_reset"] and 'scheduler' in checkpoint:
                self.scheduler.load_state_dict(
                    checkpoint['scheduler'])
            else:
                self.msg_mgr.log_warning(
                    "Restore NO Scheduler from %s !!!" % save_name)
        self.msg_mgr.log_info("Restore Parameters from %s !!!" % save_name)

    def resume_ckpt(self, restore_hint):
        if isinstance(restore_hint, int):
            save_name = self.engine_cfg['save_name']
            save_name = osp.join(
                self.save_path, 'checkpoints/{}-{:0>5}.pt'.format(save_name, restore_hint))
            self.iteration = restore_hint
        elif isinstance(restore_hint, str):
            save_name = restore_hint
            self.iteration = 0
        else:
            raise ValueError(
                "Error type for -Restore_Hint-, supported: int or string.")
        self._load_ckpt(save_name)

    def fix_BN(self):
        for module in self.modules():
            classname = module.__class__.__name__
            if classname.find('BatchNorm') != -1:
                module.eval()

    def inputs_pretreament(self, inputs):
        """Conduct transforms on input data.

        Args:
            inputs: the input data.
        Returns:
            tuple: training data including inputs, labels, and some meta data.
        """
        seqs_batch, labs_batch, typs_batch, vies_batch, seqL_batch = inputs
        seq_trfs = self.trainer_trfs if self.training else self.evaluator_trfs
        if len(seqs_batch) != len(seq_trfs):
            raise ValueError(
                "The number of types of input data and transform should be same. But got {} and {}".format(len(seqs_batch), len(seq_trfs)))
        requires_grad = bool(self.training)
        
        # Reset random seed for deterministic augmentations
        from utils import set_seed
        base_seed = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        
        if self.training:
            set_seed(base_seed + self.iteration)
        else:
            set_seed(base_seed)
        
        seqs = [np2var(np.asarray([trf(fra) for fra in seq]), requires_grad=requires_grad).float()
                for trf, seq in zip(seq_trfs, seqs_batch)]

        typs = typs_batch
        vies = vies_batch

        labs = list2var(labs_batch).long()

        if seqL_batch is not None:
            seqL_batch = np2var(seqL_batch).int()
        seqL = seqL_batch

        if seqL is not None:
            seqL_sum = int(seqL.sum().data.cpu().numpy())
            ipts = [_[:, :seqL_sum] for _ in seqs]
        else:
            ipts = seqs
        del seqs
        return ipts, labs, typs, vies, seqL

    def train_step(self, loss_sum) -> bool:
        """Conduct loss_sum.backward(), self.optimizer.step() and self.scheduler.step().

        Args:
            loss_sum:The loss of the current batch.
        Returns:
            bool: True if the training is finished, False otherwise.
        """

        self.optimizer.zero_grad()
        if loss_sum <= 1e-9:
            self.msg_mgr.log_warning(
                "Find the loss sum less than 1e-9 but the training process will continue!")

        if self.engine_cfg['enable_float16']:
            self.Scaler.scale(loss_sum).backward()
            if 'grad_clip' in self.engine_cfg and self.engine_cfg['grad_clip'] > 0:
                self.Scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.engine_cfg['grad_clip'])
            self.Scaler.step(self.optimizer)
            scale = self.Scaler.get_scale()
            self.Scaler.update()
            if scale != self.Scaler.get_scale():
                self.msg_mgr.log_debug("Training step skip. Expected the former scale equals to the present, got {} and {}".format(
                    scale, self.Scaler.get_scale()))
                return False
        else:
            loss_sum.backward()
            if 'grad_clip' in self.engine_cfg and self.engine_cfg['grad_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.engine_cfg['grad_clip'])
            self.optimizer.step()

        self.iteration += 1
        self.scheduler.step()
        return True

    def inference(self, rank):
        """Inference all the test data.

        Args:
            rank: the rank of the current process.Transform
        Returns:
            Odict: contains the inference results.
        """
        total_size = len(self.test_loader)
        if rank == 0:
            pbar = tqdm(total=total_size, desc='Transforming')
        else:
            pbar = NoOp()
        batch_size = self.test_loader.batch_sampler.batch_size
        rest_size = total_size
        info_dict = Odict()
        for inputs in self.test_loader:
            ipts = self.inputs_pretreament(inputs)
            with autocast(enabled=self.engine_cfg['enable_float16']):
                retval = self.forward(ipts)
                inference_feat = retval['inference_feat']
                for k, v in inference_feat.items():
                    inference_feat[k] = ddp_all_gather(v, requires_grad=False)
                del retval
            for k, v in inference_feat.items():
                inference_feat[k] = ts2np(v)
            info_dict.append(inference_feat)
            rest_size -= batch_size
            if rest_size >= 0:
                update_size = batch_size
            else:
                update_size = total_size % batch_size
            pbar.update(update_size)
        pbar.close()
        for k, v in info_dict.items():
            v = np.concatenate(v)[:total_size]
            info_dict[k] = v
        return info_dict

    @ staticmethod
    def run_train(model):
        """Accept the instance object(model) here, and then run the train loop."""
        try:
            from data.sampler import reset_batch_counter
            reset_batch_counter()
        except ImportError:
            pass  # Fallback if import fails
        
        for inputs in model.train_loader:
            ipts = model.inputs_pretreament(inputs)
            with autocast(enabled=model.engine_cfg['enable_float16']):
                retval = model(ipts)
                training_feat, visual_summary = retval['training_feat'], retval['visual_summary']
                del retval
            loss_sum, loss_info = model.loss_aggregator(training_feat)
            ok = model.train_step(loss_sum)
            if not ok:
                continue

            visual_summary.update(loss_info)
            visual_summary['scalar/learning_rate'] = model.optimizer.param_groups[0]['lr']

            model.msg_mgr.train_step(loss_info, visual_summary)
            if model.iteration % model.engine_cfg['save_iter'] == 0:
                # save the checkpoint
                model.save_ckpt(model.iteration)

                if model.engine_cfg['with_test']:
                    model.eval()
                    result_dict = BaseModel.run_test(model)
                    model.train()
                    if model.cfgs['trainer_cfg']['fix_BN']:
                        model.fix_BN()
                    if result_dict:
                        model.msg_mgr.write_to_tensorboard(result_dict)
                    model.msg_mgr.reset_time()
            if model.iteration >= model.engine_cfg['total_iter']:
                break

    @ staticmethod
    def run_test(model):
        """Accept the instance object(model) here, and then run the test loop."""
        try:
            from data.sampler import reset_batch_counter
            reset_batch_counter()
        except ImportError:
            pass  # Fallback if import fails
        
        evaluator_cfg = model.cfgs['evaluator_cfg']
        if torch.distributed.get_world_size() != evaluator_cfg['sampler']['batch_size']:
            raise ValueError("The batch size ({}) must be equal to the number of GPUs ({}) in testing mode!".format(
                evaluator_cfg['sampler']['batch_size'], torch.distributed.get_world_size()))
        rank = torch.distributed.get_rank()
        
        with torch.no_grad():
            info_dict = model.inference(rank)
        if rank == 0:
            loader = model.test_loader
            label_list = loader.dataset.label_list
            types_list = loader.dataset.types_list
            views_list = loader.dataset.views_list

            info_dict.update({
                'labels': label_list, 'types': types_list, 'views': views_list})

            if 'eval_func' in evaluator_cfg.keys():
                eval_func = evaluator_cfg["eval_func"]
            else:
                eval_func = 'identification'
            eval_func = getattr(eval_functions, eval_func)
            valid_args = get_valid_args(
                eval_func, evaluator_cfg, ['metric'])
            try:
                dataset_name = model.cfgs['data_cfg']['test_dataset_name']
            except:
                dataset_name = model.cfgs['data_cfg']['dataset_name']
            return eval_func(info_dict, dataset_name, **valid_args)
