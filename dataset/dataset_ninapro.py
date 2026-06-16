import glob
import re
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import scipy.io as scio
from scipy.signal import butter, filtfilt
import os

class Dataset_Ninapro(Dataset):
    def __init__(self, args, root_path, data_path='S2_E1_A1.mat', flag='train', window_size=400, step_size=100):
        self.args = args
        self.root_path = root_path
        self.data_path = data_path
        self.flag = flag
        self.window_size = window_size
        self.step_size = step_size

        self.emg_key = 'emg'
        self.label_key = 'restimulus'

        match = re.search(r'_E(\d)_', self.data_path)
        if match:
            exp_num = int(match.group(1))
            if exp_num == 1:
                self.label_range = (1, 17)
            elif exp_num == 2:
                self.label_range = (18, 40)
            elif exp_num == 3:
                self.label_range = (41, 49)
            else:
                raise ValueError(f"Unsupported experiment number: E{exp_num}")
        else:
            raise ValueError("Could not parse experiment number (E1/E2/E3) from data_path.")

        self.__read_data__()

    def __read_data__(self):
        def bandpass_filter(signal, low=20, high=450, fs=2000, order=4):
            nyq = 0.5 * fs
            low /= nyq
            high /= nyq
            b, a = butter(order, [low, high], btype='band')
            return filtfilt(b, a, signal, axis=0)

        all_emg = []
        all_labels = []

        all_files = []
        for part in self.data_path.split(','):
            part = part.strip()
            matched = sorted(glob.glob(os.path.join(self.root_path, part)))
            if len(matched) == 0:
                print(f"Warning: no files matched pattern {part}")
            all_files.extend(matched)

        assert len(all_files) > 0, f"No files matched any of: {self.data_path}"
        file_paths = all_files

        for file in file_paths:
            mat = scio.loadmat(file)
            if self.emg_key not in mat or self.label_key not in mat:
                print(f"Skipped {file}: missing keys")
                continue

            emg = mat[self.emg_key]
            labels = mat[self.label_key].squeeze()

            low, high = self.label_range
            valid_idx = (labels >= low) & (labels <= high)
            emg = emg[valid_idx]
            labels = labels[valid_idx]
            if emg.shape[0] < 27 or len(labels) == 0:
                continue

            emg_filtered = bandpass_filter(mat[self.emg_key][valid_idx])

            emg = emg_filtered
            labels = labels - low
            all_emg.append(emg)
            all_labels.append(labels)

        emg_all = np.concatenate(all_emg, axis=0)
        labels_all = np.concatenate(all_labels, axis=0)
        mean = np.mean(emg_all, axis=(0,), keepdims=True)
        std = np.std(emg_all, axis=(0,), keepdims=True) + 1e-5
        emg_all = (emg_all - mean) / std

        def window_samples(emg_data, label_data):
            samples, labels = [], []
            T = emg_data.shape[0]
            for i in range(0, T - self.window_size, self.step_size):
                x_win = emg_data[i:i + self.window_size]
                y_win = label_data[i:i + self.window_size]
                y = y_win[len(y_win) // 2]
                samples.append(x_win)
                labels.append(y)
            return np.array(samples), np.array(labels)

        X_all, y_all = window_samples(emg_all, labels_all)

        X_temp, X_test, y_temp, y_test = train_test_split(
            X_all, y_all, test_size=0.2, stratify=y_all, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.125, stratify=y_temp, random_state=42)

        if self.flag == 'train':
            self.X, self.Y = X_train, y_train
        elif self.flag == 'val':
            self.X, self.Y = X_val, y_val
        else:
            self.X, self.Y = X_test, y_test

        self.norm_mean = mean
        self.norm_std = std
        self.num_classes = self.label_range[1] - self.label_range[0] + 1
        print(f"{self.flag} set size: {len(self.X)} | num_classes: {self.num_classes}")

    def __getitem__(self, index):
        seq_x = self.X[index]
        seq_y = self.Y[index]

        return torch.tensor(seq_x, dtype=torch.float32), torch.tensor(seq_y, dtype=torch.long)

    def __len__(self):
        return len(self.X)
