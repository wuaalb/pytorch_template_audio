import os
import argparse
from pathlib import Path

import torch
import soundfile as sf

from train import create_model
from utils import count_parameters, load_checkpoint_inference, find_latest_checkpoint
import hparams


def predict(args):
    inp_dir = Path(args.inp_dir)
    out_dir = Path(args.out_dir)

    # set up cuda
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'#str(hparams.gpu)  # comma separated for multi-gpu
    torch.backends.cudnn.enabled = True
    #torch.backends.cudnn.benchmark = True  # faster

    device = torch.device('cuda')

    # find input files
    fns_wav = inp_dir.glob('*.wav')
    fns_wav = list(fns_wav)

    if len(fns_wav) == 0:
        raise IOError('No input files found')

    print('Num. files: {:d}'.format(len(fns_wav)))

    # create model
    print('Creating model...')
    model = create_model()
    print('Num. parameters: {:,d}'.format(count_parameters(model)))

    model.to(device)

    # get checkpoint dir
    if args.resume_dir is not None:
        ckptdir = Path(args.resume_dir)
    else:
        ckptdir = Path('./experiments/{}/checkpoints'.format(hparams.experiment_tag))

    # get checkpoint step
    resume_step = args.resume_step
    if resume_step < 0:
        fn_checkpoint = find_latest_checkpoint(ckptdir, '*.pt')
    else:
        fn_checkpoint = ckptdir / 'model.ckpt-{:d}.pt'.format(resume_step)

    # load checkpoint
    print('Loading checkpoint {}...'.format(fn_checkpoint))
    global_step = load_checkpoint_inference(fn_checkpoint, model, not args.resume_non_ema)
    print('Step: {:,d}'.format(global_step))

    # create output directory
    out_dir.mkdir(parents=True, exist_ok=True)

    # process files
    print('Processing files...')
    model.eval()  # inference mode
    with torch.no_grad():
        for fn_wav in fns_wav:
            print(fn_wav.stem)

            # load wav
            x, sr = sf.read(fn_wav, always_2d=True, dtype='float32')
            x = x.T  # TC -> CT
            sr = float(sr)
            assert sr == hparams.sr

            # XXX: could resample (downsample) here for convenience, instead of simply asserting sample rate is the one we expect

            # to tensor
            x = x[None, :, :]  # CT -> NCT
            x = torch.from_numpy(x)
            x = x.to(device)

            # process
            x = model(x)

            # to numpy
            x = x.cpu().numpy()
            x = x[0, :, :]  # NCT -> CT
            x = x.T  # CT -> TC

            # save wav
            fn_out_wav = out_dir / '{}.wav'.format(fn_wav.stem)
            sf.write(fn_out_wav, x, int(sr), subtype='FLOAT')


    print('OK')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inp-dir', type=str, default='./inference/inp', help='Path to input files to process')
    parser.add_argument('--out-dir', type=str, default='./inference/out', help='Path to output files')
    parser.add_argument('--resume-dir', type=str, default=None, help='Path to the model checkpoint to restore (if not given use last checkpoint from experiment in hparams)')
    parser.add_argument('--resume-step', type=int, default=-1, help='Step of the model checkpoint restore (if not given use last checkpoint from experiment in hparams)')
    parser.add_argument('--resume-non-ema', action='store_true', help='Restore non-EMA weights from checkpoint')
    args = parser.parse_args()
    predict(args)

if __name__ == '__main__':
    main()