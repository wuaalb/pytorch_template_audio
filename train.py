import os
import random
from pathlib import Path
import time
import datetime
from collections import defaultdict
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from model import Model
from dataset import load_file_list, WavDataset
from audio import MelSpec
from utils import backup_source, worker_init_fn, count_parameters, ExpDecayWarmupScheduler, EMA, save_checkpoint, load_checkpoint, find_latest_checkpoint
import hparams


def create_model():
    model = Model(hparams.n_mel)
    return model

def train(args):
    # set up cuda
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(hparams.gpu)  # comma separated for multi-gpu
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True  # faster

    device = torch.device('cuda')

    # seed random
    random.seed(hparams.seed)
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed_all(hparams.seed)  # all GPUs

    # set up folders
    dsetdir = Path(hparams.dsetdir)
    outdir = Path('./experiments/{}'.format(hparams.experiment_tag))
    ckptdir = outdir / 'checkpoints'
    logdir = outdir / 'log'
    outdir.mkdir(parents=True, exist_ok=True)
    ckptdir.mkdir(parents=True, exist_ok=True)
    logdir.mkdir(parents=True, exist_ok=True)

    # backup source code
    backup_source(outdir / 'source.zip', ['./*.py'])

    # create model
    model = create_model()
    print('Num. parameters: {:,d}'.format(count_parameters(model)))

    model.to(device)

    # dataset
    file_list_trn = load_file_list(dsetdir / 'filelist_trn.txt')
    print('Train set: {:,d} examples'.format(len(file_list_trn)))
    dset_trn = WavDataset(file_list_trn, hparams.seq_dur, hparams.sr)
    loader_trn = torch.utils.data.DataLoader(dset_trn, batch_size=hparams.batch_size, shuffle=True, drop_last=True, num_workers=hparams.n_workers, worker_init_fn=worker_init_fn, pin_memory=True)
    # XXX: drop_last=True because code below for averaging losses assumes all steps to have equal weight (minor issue)
    
    file_list_val = load_file_list(dsetdir / 'filelist_val.txt')
    print('Validation set: {:,d} examples'.format(len(file_list_val)))
    dset_val = WavDataset(file_list_val, hparams.seq_dur, hparams.sr)
    loader_val = torch.utils.data.DataLoader(dset_val, batch_size=hparams.batch_size, shuffle=False, drop_last=True, num_workers=hparams.n_workers, worker_init_fn=worker_init_fn, pin_memory=True)
    # XXX: also drop_last=True for simplicity, however as here shuffle=False, the last partial batch of the validation set is "lost"

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=hparams.learning_rate, **hparams.opt_params)

    # lr schedule
    if hparams.lr_decay > 0:
        lr_scheduler = ExpDecayWarmupScheduler(optimizer, warmup_steps=hparams.lr_decay_step_size, step_size=hparams.lr_decay_step_size, decay=hparams.lr_decay)
    else:
        lr_scheduler = None

    # EMA
    if hparams.ema_decay > 0:
        model_ema = create_model()
        model_ema.to(device)
        ema = EMA(model, model_ema, hparams.ema_decay, hparams.ema_warmup_steps)
    else:
        ema = None

    # melspec
    melspec_fn = MelSpec(hparams.sr, hparams.hop_len, hparams.win_len, hparams.fft_len, hparams.window_kind,
                         hparams.n_mel, hparams.fmin, hparams.fmax, norm_window=True, filter_norm=1, center=True, pad_mode='reflect', clamp_min_db=hparams.clamp_min_db, to_db=True)

    melspec_fn.to(device)

    # logging
    log_writer = SummaryWriter(logdir, flush_secs=30)

    # load checkpoint for resuming training
    if args.resume_dir is not None:
        resume_dir = Path(args.resume_dir)
        resume_step = args.resume_step
        if resume_step < 0:
            fn_checkpoint = find_latest_checkpoint(resume_dir, '*.pt')
        else:
            fn_checkpoint = resume_dir / 'model.ckpt-{:d}.pt'.format(resume_step)

        print('Resuming training from checkpoint {}...'.format(fn_checkpoint))
        global_step = load_checkpoint(fn_checkpoint, model, model_ema, optimizer, lr_scheduler)

        if hparams.ema_decay > 0:
            ema = EMA(model, model_ema, hparams.ema_decay, hparams.ema_warmup_steps)
    else:
        global_step = 0  # train from scratch

    # helper to compute loss function
    # reused between training and validation
    def compute_loss(x, model):
        # compute mel-spectrogram
        x = melspec_fn(x)

        # compute loss
        y = model(x)
        loss = F.mse_loss(y, x)

        return loss

    # loop
    # XXX: this loop prints progress per update, validate every each epoch, save checkpoint every N updates, write average loss every M updates
    #      alternatively could e.g. validate after every epoch, print (averaged) progress every epoch, etc. or write log every update, etc.
    #      this mostly depends on the size of the dataset/epoch, and the time each update takes
    #      here an 'epoch' doesn't see the entire dataset, just a random excerpt from each file on disk
    time_used_total = 0.0
    losses = defaultdict(float)
    stop = False
    while True:
        if stop:
            break

        for x in loader_trn:
            if global_step >= hparams.n_steps:
                stop = True
                break  # stop training

            start = time.perf_counter()

            x = x.to(device)

            loss = compute_loss(x, model)

            # optimization step
            optimizer.zero_grad()
            loss.backward()

            if hparams.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), hparams.clip_grad_norm)

            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            if ema is not None:
                ema.step(global_step)

            # time used
            time_used = time.perf_counter() - start
            time_used_total += time_used

            # next step
            # NOTE: do BEFORE using global_step (logging, saving checkpoints, ..)
            global_step += 1

            # log results for step
            print('step {:d} --loss {:.5f} --time {:.2f}'.format(global_step, loss.item(), time_used))

            # save checkpoint
            if global_step % hparams.checkpoint_step == 0:
                print('saving checkpoint...')
                fn_checkpoint = ckptdir / 'model.ckpt-{:d}.pt'.format(global_step)
                save_checkpoint(fn_checkpoint, model, model_ema, optimizer, lr_scheduler, global_step)

            # save log
            losses['loss'] += loss.item()/hparams.summary_step
            if global_step % hparams.summary_step == 0:
                for key in losses:
                    log_writer.add_scalar(key, losses[key], global_step)
                losses = defaultdict(float)  # reset to zero

        # validate
        if ema is not None:
            model_eval = model_ema
        else:
            model_eval = model
        model_eval.eval()
        val_losses = defaultdict(float)
        val_n = 0
        with torch.no_grad():
            for x in loader_val:  # XXX: add e.g. tqdm for big validation sets
                x = x.to(device)

                loss = compute_loss(x, model_eval)

                val_losses['loss_val'] += loss.item()
                val_n += 1
        model_eval.train()
        print('VALIDATION step {:d} --loss {:.5f}'.format(global_step, val_losses['loss_val']/val_n))
        for key in val_losses:
            log_writer.add_scalar(key, val_losses[key]/val_n, global_step)

    print('Total training time: {}'.format(str(datetime.timedelta(seconds=time_used_total))))

    print('OK')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume-dir', type=str, default=None, help='Path to the model checkpoint to restore')
    parser.add_argument('--resume-step', type=int, default=-1, help='Step of the model checkpoint restore')
    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()
