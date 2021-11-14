import os
import torch.utils.data
from glob import glob
import random
import numpy as np
from torchaudio.transforms import MuLawEncoding


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
