import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np 
from librosa.util import find_files
from torchaudio import load
from torch import nn
import os 
import re
import random
import torchaudio
import sys
import time
import glob
import tqdm
from pathlib import Path


class AccentClassifiDataset(Dataset):
    def __init__(self, mode, file_path, max_timestep=None):
        self.root = file_path
        file_path_mode = os.path.join(self.root, mode)
        key_file_path = os.path.join(file_path_mode, "key.lst")

        dataset, accents = list(), list()
        count = 0
        
        audio_dir = os.path.join(file_path_mode, "audio")

        with open(key_file_path, "r") as file:
            for line in file.readlines():
                name, accent = line.split()
                path = os.path.join(audio_dir, name + ".wav")
                if os.path.isfile(path):
                    count +=1
                
                dataset.append(path)
                accents.append(accent)
      
        assert count == len(dataset)
        print(f'[AccentClassifiDataset] - there are {len(dataset)} files found for {mode}.')

        self.max_timestep = max_timestep
        self.dataset = dataset
        self.label, self.num_accents, self.label2accent_dict = self.build_label(accents)

    def build_label(self, accents):

        unique_accent = sorted(set(accents))

        accent_to_number = {accent: i for i, accent in enumerate(unique_accent)}
         
        accents_numbers = [accent_to_number[accent] for accent in accents]

        label2accent_dict = {number: accent for accent, number in accent_to_number.items()}

        return accents_numbers, len(unique_accent), label2accent_dict
    
    @classmethod
    def label2accent(cls, labels, label2accent_dict):
        return [label2accent_dict[label] for label in labels]
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        wav, sr = torchaudio.load(self.dataset[idx])
        wav = wav.squeeze(0)
        length = wav.shape[0]

        if self.max_timestep !=None:
            if length > self.max_timestep:
                start = random.randint(0, int(length-self.max_timestep))
                wav = wav[start:start+self.max_timestep]
                length = self.max_timestep

        def path2name(path):
            return Path("-".join((Path(path).parts)[-3:])).stem

        path = self.dataset[idx]
        return wav.numpy(), self.label[idx], path2name(path)
        
    def collate_fn(self, samples):
        return zip(*samples)
