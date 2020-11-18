# experiment, hardware
experiment_tag = 'run001-wip'
gpu = 0
n_workers = 2  # NOTE: when > 0, code cannot be debugged or edited while being executed
seed = 1234

# dataset
dsetdir = '/my_dataset'

# acoustic parameters
sr = 44100.0
hop_len = 256  # XXX: normally hop_len, win_len, etc. function of sr
win_len = 2048
fft_len = 2048
window_kind = 'hann'
n_mel = 128
fmin = 40.0
fmax = 12000.0
clamp_min_db = -120.0

# optimizer
seq_dur = 0.1#1.0
batch_size = 32
learning_rate = 1e-4
opt_params = {'betas': (0.9, 0.999), 'eps': 1e-8}  # torch.optim.Adam defaults
lr_decay_step_size = 200_000  # lr fixed for first step_size steps, then drop down by factor, then (continuously) decays by factor every step_size steps
lr_decay = 0.5
clip_grad_norm = None
n_steps = 1_000_000
#ema_decay = 0  # disable EMA
ema_decay = 0.999
ema_warmup_steps = 1_000
checkpoint_step = 5_000
summary_step = 500
