from pathlib import Path

import numpy as np
import torch
import soundfile as sf


# load file list text file
# comments are escaped using '#'
# file names should be relative to path of the file list
# in case each example corresponds to multiple files, this can be handled by
# omitting extensions in the file list or using only the 'root' part of the file names,
# and let the Dataset load the corresponding files with different extensions and/or suffixes
def load_file_list(fn_filelist_txt):
    fn_filelist_txt = Path(fn_filelist_txt)
    root_dir = fn_filelist_txt.parent

    examples = []
    with open(fn_filelist_txt, 'r') as f:
        for ln in f:
            # handle comment
            b_comment = ln.find('#')
            if b_comment >= 0:
                ln = ln[:b_comment]
                
            # strip white space
            ln = ln.strip()

            # skip empty lines (e.g. entire line is comment)
            if len(ln) == 0:
                continue

            # add example to list
            example = str(root_dir / ln)
            examples.append(example)
    
    return examples


class WavDataset(torch.utils.data.Dataset):
    # file_list list (full filename without extension)
    def __init__(self, file_list, seq_dur, sr, on_too_short='raise'):
        super().__init__()
        self.file_list = file_list
        self.seq_len = int(np.rint(seq_dur*sr))
        self.sr = sr
        self.on_too_short = on_too_short

    def __getitem__(self, index):
        fn_base = self.file_list[index]
        fn_wav = '{}.wav'.format(fn_base)

        # XXX: here, we read the entire file from disk and then do random crop
        # alternatively we could use pysoundfile.read()'s start/stop arguments
        # we assume that all files are relatively short, and reading the entire
        # probably makes caching disk i/o easier for the OS

        # read file
        x, sr = sf.read(fn_wav, always_2d=True, dtype='float32')
        n = len(x)
        x = x.T  # TC -> CT
        sr = float(sr)
        assert sr == self.sr

        # random crop
        if n < self.seq_len:
            if self.on_too_short == 'raise':
                # XXX: if file is shorter than request file length, just raise exception
                # assuming should files are already filtered from the file list when preprocessing the dataset
                # alternatively, short files can be zero padded, but this may have some (minor) side effects (affecting loss, gradients, etc.)
                raise IOError('file too short for requested training sequence length; pre-filter file list')
            elif self.on_too_short == 'pad':
                # XXX: maybe in different cases padding should be left/right/centered
                raise NotImplementedError('file too short for requested training sequence length; implement padding')
            else:
                raise ValueError('invalid on_too_short')
        else:
            b = np.random.randint(0, n - self.seq_len + 1)  # [lo, hi[
            e = b + self.seq_len
            x = x[:, b:e]

        # to pytroch tensor
        x = torch.from_numpy(x)

        return x

    def __len__(self):
        return len(self.file_list)

