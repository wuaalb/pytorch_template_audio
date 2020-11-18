from pathlib import Path
import warnings
import torch
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np


# find checkpoint with latest creation time (meta data modification on unix)
def find_latest_checkpoint(path, pattern='*.pt'):
    path = Path(path)
    fns = path.glob(pattern)
    fns = list(fns)
    if len(fns) == 0:
        return None
    fn_latest = max(fns, key=lambda fn: fn.stat().st_ctime)
    return fn_latest

# raise if inconsistent
def check_step_consistency(optimizer, lr_scheduler, step):
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad == None:
                continue
            if optimizer.state[p]['step'] != step:
                raise IOError('optimizer step and loop step inconsistent')

    if lr_scheduler:
        if lr_scheduler.last_epoch != step:
            raise IOError('lr scheduler step and loop step inconsistent')

def save_checkpoint(fn_checkpoint, model, model_ema, optimizer, lr_scheduler, step):
    # check step consistency
    check_step_consistency(optimizer, lr_scheduler, step)
    
    # get data from components
    checkpoint_dict = {}
    checkpoint_dict['model'] = model.state_dict()
    if model_ema is not None:
        checkpoint_dict['model_ema'] = model_ema.state_dict()
    checkpoint_dict['optimizer'] = optimizer.state_dict()
    if lr_scheduler is not None:
        checkpoint_dict['lr_scheduler'] = lr_scheduler.state_dict()
    checkpoint_dict['step'] = step
    
    # save file
    torch.save(checkpoint_dict, fn_checkpoint)

def load_checkpoint(fn_checkpoint, model, model_ema, optimizer, lr_scheduler):
    # load file
    checkpoint_dict = torch.load(fn_checkpoint)

    # load data into components
    model.load_state_dict(checkpoint_dict['model'])
    if model_ema is not None:
        model_ema.load_state_dict(checkpoint_dict['model_ema'])
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
    step = checkpoint_dict['step']

    # check step consistency
    check_step_consistency(optimizer, lr_scheduler, step)

    return step


# e.g. backup_source(out_dir / 'source.zip', ['./*.py', './models/*.py'])
def backup_source(fn_out_zip, path_pattern_list):
    from pathlib import Path
    from zipfile import ZipFile
    with ZipFile(fn_out_zip, 'w') as backup:
        for path_pattern in path_pattern_list:
            path_pattern = Path(path_pattern)
            dir = path_pattern.parent
            pattern = path_pattern.name
            fns_py = list(dir.glob(pattern))
            if len(fns_py) == 0:
                raise IOError('No files found to backup!')
            for fn_py in fns_py:
                backup.write(fn_py)


def worker_init_fn(wid):
    np.random.seed(torch.initial_seed() % (2 ** 32))  # seed numpy with worker-specific seed from pytorch, to avoid all workers using same random seed (afaik only problem on linux, not windows); note no need to add worker id to torch.initial_seed() (already worker specific), but have to wrap to uint32 for numpy


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ExpDecayWarmupScheduler(_LRScheduler):
    # initial lr for n_steps_warmup steps (updates),
    # then drops to initial lr * decay, and continues (smoothly) exponentially decaying by factor decay every step_size steps
    # continue decaying until lr_floor is reached, then keep constant
    def __init__(self, optimizer, warmup_steps=200_000, step_size=200_000, decay=0.5, lr_floor=1e-6):
        self.warmup_steps = warmup_steps
        self.step_size = step_size
        self.decay = decay
        self.lr_floor = lr_floor
        super().__init__(optimizer)  # calls self.get_lr(), thus do last

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        ## boilerplate from torch.optim (1.5)

        last_step = self.last_epoch  # _LRScheduler (originally) designed for epoch-wise scheduling; incremented BEFORE get_lr() is called and initialized to -1
        if last_step >= self.warmup_steps:
           lr_scale = self.decay**(1 + (last_step - self.warmup_steps)/self.step_size)
        else:
           lr_scale = 1.0
        # XXX: get_lr() in torch.optim (1.5) tend to continuously update param_group['lr'] instead of using base_lrs
        #      possibly related to chainable lr schedulers

        return [max(lr_scale*base_lr, self.lr_floor) for base_lr in self.base_lrs]


# XXX:
# had a more advanced version, but didn't seem to work properly (didn't debug)
# improvements/differences
# - just pass source model, create internal target model via deep_copy()
# - internal step counter, instead of passing step counter on step() for warmup
# - state_dict(), load_state_dict() to store/load internal target model and step counter
# - optional device argument, so EMA (target) model can be kept on CPU model rather than using up GPU model
# - optional list of excluded parameter names (e.g. don't use up compute decaying, some big constant tensors stored in state_dict using register_buffer())
# - switching target model to eval mode (assuming it will only ever be used for inference or keeping track of EMA weights, not training directly)
# - set requires_grad_(False) on all parameters of internal target model (not just make operations have no gradients, but the actual parameters themselves)
class EMA(object):
    def __init__(self, source, target, decay, warmup_steps=0):
        self.source = source
        self.target = target
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.source_dict = self.source.state_dict()
        self.target_dict = self.target.state_dict()
        with torch.no_grad():
            for key in self.source_dict:
                self.target_dict[key].data.copy_(self.source_dict[key].data)

    # pass step_idx if warup_steps > 0
    # XXX: need to pass it explicitly, because EMA is state-less and thus cannot have internal step counter which
    # is properly stored/loaded from checkpoint
    def step(self, step_idx=None):
        if step_idx and step_idx < self.warmup_steps:
            decay = 0.0  # no EMA, use source parameters directly
        else:
            decay = self.decay
        with torch.no_grad():
            for key in self.source_dict:
                self.target_dict[key].data.copy_(self.target_dict[key].data*decay + self.source_dict[key].data*(1.0 - decay))