import os
import sys
import torch.utils.data
from glob import glob
import random
import numpy as np
from torchaudio.transforms import MuLawEncoding

sys.path.insert(0, 'tacotron2')
from tacotron2.layers import TacotronSTFT
sys.path.remove('tacotron2')

class AudioData(torch.utils.data.Dataset):
    def __init__(self, source_pass, train=True):
        super(AudioData, self).__init__()
        self.files = sorted(glob(os.path.join(source_pass, '*.npy')))
        self.stride = 81920
        if train:
            self.files = self.files[:27000]
        else:
            self.files = self.files[27000:30000]

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        head1, head2 = random.sample(range(len(data) - self.stride), 2)
        data = torch.from_numpy(data[head1:head1 + self.stride]).unsqueeze(0)
        data = MuLawEncoding(256)(data).float()
        return data

    def __len__(self):
        return len(self.files)

class MelData(torch.utils.data.Dataset):
    def __init__(self, source_pass, train=True):
        super(MelData, self).__init__()
        self.files = sorted(glob(os.path.join(source_pass, '*.npy')))
        self.stride = 81920
        if train:
            self.files = self.files[:11000]
        else:
            self.files = self.files[11000:]
        self.stft = TacotronSTFT(filter_length=1024,
                                 hop_length=256,
                                 win_length=1024,
                                 sampling_rate=16000,
                                 mel_fmin=0.0, mel_fmax=8000.0)

    def get_mel(self, audio):
        audio_norm = audio.unsqueeze(0)
        audio_norm = torch.clamp(audio_norm, -1., 1.)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)[:, :-1]
        return melspec

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        head1, head2 = random.sample(range(len(data) - self.stride), 2)
        data = data[head1:head1 + self.stride]
        data = torch.from_numpy(data).float()
        return self.get_mel(data)

    def __len__(self):
        return len(self.files)
