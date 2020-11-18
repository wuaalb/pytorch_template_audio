from pathlib import Path

import numpy as np
import torch
import soundfile as sf


# load file list text file
# comments are escaped using '#'
# files with multiple columns require a mandatory initial comment listing names of columns (e.g. '# mix vocals')
# columns are separated by '\t' (not spaces), so file names can have spaces
# file names should be relative to path of file list, and not have an extension
# returns list of dicts (multicol=True) or list (multicol=False)
def load_file_list(fn_filelist_txt, multicol=True):
    fn_filelist_txt = Path(fn_filelist_txt)
    root_dir = fn_filelist_txt.parent

    examples = []  # list of dicts, or list
    with open(fn_filelist_txt, 'r') as f:
        # read lines
        lns = [ln.strip() for ln in f.readlines()]

        # read header
        if multicol:
            header = lns[0]
            lns = lns[1:]
            if not header.startswith('#'):
                raise IOError('file list should have comment header containing instrument names')
            header = header[1:].strip()  # strip escape and leading (and trailing) whitespace
            #instruments = header.split()  # NOTE: assume instruments are separated by whitespace (that is, whitespace is not allowed in instrument names)
            instruments = header.split('\t')  # NOTE: assume instruments are separated by tab (that is, spaces are allowed in instrument names)

        # read file list
        for ln in lns:  # header already skipped above
            # handle comment
            b_comment = ln.find('#')
            if b_comment >= 0:
                ln = ln[:b_comment].strip()

            if len(ln) == 0:
                continue  # entire line is comment

            # split columns, convert to full filenames
            if multicol:
                #cols = ln.split()  # NOTE: assume files are separated by whitespace (that is, whitespace is not allowed in filenames)
                cols = ln.split('\t')  # NOTE: assume instruments are separated by tab (that is, spaces are allowed in filenames)

                if len(cols) != len(instruments):
                    raise IOError('number of columns in line does not match header')

                cols = [str(root_dir / col) for col in cols]  # full name without extension

                example = dict(zip(instruments, cols))
                examples.append(example)
            else:
                example = str(root_dir / ln)
                examples.append(example)
    
    return examples


class WavDataset(torch.utils.data.Dataset):
    # file_list list (full_filename_no_ext), or list of dicts (instrument_name: full_filename_no_ext)
    def __init__(self, file_list, seq_dur, sr, on_too_short='raise'):
        super().__init__()
        self.file_list = file_list
        self.seq_len = int(np.rint(seq_dur*sr))
        self.sr = sr
        self.on_too_short = on_too_short

    def __getitem__(self, index):
        file = self.file_list[index]
        if isinstance(file, dict):
            return {instrument: self.read_file_and_crop(file[instrument]) for instrument in file.keys()}
        else:
            return self.read_file_and_crop(file)

    def read_file_and_crop(self, fn_base):
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

