import torch
import torch.nn as nn
import numpy as np
import librosa


# samples to frames
def get_n_frames_spec(n, hop_len, win_len, fft_len, center):
    if center:
        n0 = fft_len//2 + n + fft_len//2  # center padding
    else:
        n0 = n
    n_frames = 1 + (n0 - win_len)//hop_len
    return n_frames

# frames to samples
def get_n_samples_spec(n_frames, hop_len, win_len, fft_len, center):
    n0 = (n_frames - 1)*hop_len + win_len  # n0 .. n0 + hop_len will result in n_frames
    if center:
        n = n0 - fft_len
    else:
        n = n0
    return n


class Spec(nn.Module):
    def __init__(self, hop_len, win_len, fft_len, window_kind, center=True, pad_mode='reflect'):
        super().__init__()

        window = librosa.filters.get_window(window_kind, win_len, fftbins=True)
        window = torch.from_numpy(window).float()
        self.register_buffer('window', window)  # include in module's state (checkpoints, to device, ..)

        self.hop_len = hop_len
        self.win_len = win_len
        self.fft_len = fft_len
        self.center = center
        self.pad_mode = pad_mode

    # stft
    def forward(self, x):
        batch_size, n_chan, n_timesteps = x.size()
        x = x.reshape(batch_size*n_chan, n_timesteps)  # N,C,T -> N*C,T

        X = torch.stft(x, n_fft=self.fft_len, hop_length=self.hop_len, win_length=self.win_len, window=self.window, center=self.center, pad_mode=self.pad_mode, normalized=False, onesided=True)

        X = X.reshape(batch_size, n_chan, *X.shape[-3:])  # N*C,H,W,2 -> N,C,H,W,2  (H=fft_len//2+1, W=T, 2->complex)

        return X

    # istft
    def reverse(self, X):
        batch_size, n_chan, spec_size, n_timesteps, two = X.size()
        assert two == 2
        X = X.reshape(batch_size*n_chan, *X.shape[-3:])  # N,C,H,W,2 -> N*C,H,W,2

        x = torch.istft(X, n_fft=self.fft_len, hop_length=self.hop_len, win_length=self.win_len, window=self.window, center=self.center, normalized=False, onesided=True, length=None)

        x = x.reshape(batch_size, n_chan, -1)

        return x


# pytorch module to compute mel-spec from waveform
# NOTE
# - by default output is linear melspec (not decibels)
# - by default output is not clipped to some minimum value (to avoid taking log of zero); however, this is needed when gradients are required (to avoid div zero in gradient of sqrt)
# - by default normalizes stft window by 2/sum(window)  [not default of e.g. librosa]
# - by default uses l1 normalization of discrete frequency mel-scale filters  [not default of e.g. librosa]
class MelSpec(nn.Module):
    def __init__(self, sr, hop_len, win_len, fft_len, window_kind, n_mel, fmin, fmax, norm_window=True, filter_norm=1, center=True, pad_mode='reflect', clamp_min_db=None, to_db=False):
        super().__init__()

        window = librosa.filters.get_window(window_kind, win_len, fftbins=True)
        if norm_window:
            window *= 2/np.sum(window)  # normalize window
        mel_basis = mel_filters(sr, fft_len, n_mel, fmin, fmax, filter_norm)
        window = torch.from_numpy(window).float()
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer('window', window)  # include in module's state (checkpoints, to device, ..)
        self.register_buffer('mel_basis', mel_basis)

        self.fft_len = fft_len
        self.hop_len = hop_len
        self.win_len = win_len
        self.center = center
        self.pad_mode = pad_mode

        self.clamp_min = None
        if clamp_min_db is not None:
            self.clamp_min = 10**(clamp_min_db/10)  # to clamp pspec

        self.to_db = to_db

    def forward(self, x):
        n_chan = x.size(1)
        x = x.reshape(x.size(0)*x.size(1), x.size(2))  # NCT -> N*C,T
        spec = torch.stft(x, self.fft_len, self.hop_len, self.win_len, self.window, self.center, self.pad_mode, normalized=False, onesided=True)
        real, imag = spec.unbind(-1)
        pspec = real**2 + imag**2
        if self.clamp_min is not None:
            pspec = torch.clamp(pspec, min=self.clamp_min)
        magspec = torch.sqrt(pspec)
        melspec = torch.matmul(self.mel_basis, magspec)
        if self.to_db:
            melspec = 20*torch.log10(melspec)
        melspec = melspec.reshape(melspec.size(0)//n_chan, melspec.size(1)*n_chan, melspec.size(2))  # N*Cin,Cmel,T -> N,Cin*Cmel,T
        return melspec


# wrapper around librosa.filter.mel for backwards compatibility
# NOTE: for librosa 0.8.0+ (not released; April 2020) we can just use librosa.filters.mel(.., norm=filter_norm)
from pkg_resources import parse_version  # for backwards compatibility
def mel_filters(sr, fft_len, n_mel, fmin, fmax, filter_norm):
    # 'slaney' -> 'slaney' (>= 0.7.2)
    # 'slaney' -> 1 (< 0.7.2)
    # 1 -> None (any, normlize ourselves)
    if filter_norm == 'slaney':
        if parse_version(librosa.__version__) < parse_version('0.7.2'):
            filter_norm_ = 1  # slaney in < 0.7.2
        else:
            filter_norm_ = 'slaney'
    elif filter_norm == 1:
        filter_norm_ = None  # normalize ourselves
    else:
        raise ValueError('invalid filter_norm')

    mel_basis = librosa.filters.mel(sr, fft_len, n_mel, fmin, fmax, norm=filter_norm_)

    if filter_norm == 1:
        mel_basis = librosa.util.normalize(mel_basis, norm=1, axis=1)

    return mel_basis

# compute actual stft parameters from natural parameters,
# extracted to function to ensure coherence
# can be called with scalars or lists/arrays
def get_stft_params(sr, hoptime, wintime, fft_len_min):
    hoptime = np.array(hoptime)
    wintime = np.array(wintime)
    fft_len_min = np.array(fft_len_min)

    hop_len = np.round(hoptime*sr).astype(int)
    win_len = np.round(wintime*sr).astype(int)
    fft_len = (2**np.ceil(np.log2(win_len))).astype(int)  # next power-of-2
    fft_len = np.maximum(fft_len_min, fft_len)  # minimum size, e.g. to avoid mel filter bank having empty filters

    if hop_len.ndim == 0:
        hop_len = int(hop_len)
    if win_len.ndim == 0:
        win_len = int(win_len)
    if fft_len.ndim == 0:
        fft_len = int(fft_len)

    return hop_len, win_len, fft_len

# assumes input is linear melspec, without clamping
def lin_melspec_to_norm_log_melspec(melspec, min_level_mel_db, max_level_mel_db):
    # clip low values
    min_level_mel = 10**(min_level_mel_db/20)
    melspec = torch.clamp(melspec, min=min_level_mel)

    # to decibels
    melspec = 20*torch.log10(melspec)

    # normalize [min_level_mel_db, max_level_mel_db] -> [0, 1]
    melspec = (melspec - min_level_mel_db)/(max_level_mel_db - min_level_mel_db)
    
    return melspec

def norm_log_melspec_to_lin_melspec(melspec, min_level_mel_db, max_level_mel_db):
    melspec = melspec*(max_level_mel_db - min_level_mel_db) + min_level_mel_db
    melspec = 10**(melspec/20)
    return melspec


