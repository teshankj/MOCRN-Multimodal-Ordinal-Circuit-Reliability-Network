"""
Flexible Parser for Circuit Waveform Data
Auto-detects number of channels and statistical features from file
"""

import numpy as np
import re
from typing import List, Tuple, Dict
import torch
from torch.utils.data import Dataset

def parse_flexible_waveform_data(
    filepath: str,
    max_len: int = 200,
    verbose: bool = True
) -> Tuple[np.ndarray, List[str], Dict]:
    """
    Parse waveform data - handles concatenated headers like: timeV(in)V(n1)V(n2)
    """

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    if verbose:
        print(f"\nReading file: {filepath}")

    samples = []
    current_sample = None
    step_pattern = re.compile(r'Step Information: Ft=(\d+) Deg=(\d+) Run=(\d+)')
    global_channel_names = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check for step information
        step_match = step_pattern.search(line)
        if step_match:
            if current_sample is not None and len(current_sample.get('data', [])) > 0:
                samples.append(current_sample)

            ft, deg, run = map(int, step_match.groups())
            current_sample = {
                'fault_type': ft,
                'degradation': deg,
                'run': run,
                'data': []
            }
            continue

        if 'time' in line.lower() and ('V(' in line or 'I(' in line):
            signal_pattern = re.compile(r'([VI])\(([^)]+)\)', re.IGNORECASE)
            matches = signal_pattern.findall(line)
            if matches:
                global_channel_names = [f"{sig[0].upper()}({sig[1]})" for sig in matches]
                if verbose:
                    print(f"✓ Found {len(global_channel_names)} channels: {global_channel_names}")
                continue


        if current_sample is not None and not line.startswith('Step'):
            try:
                num_pattern = r'[+-]?\d+\.?\d*[eE]?[+-]?\d+'
                numbers = re.findall(num_pattern, line)
                if len(numbers) > 1:
                    values = [float(x) for x in numbers[1:]]
                    current_sample['data'].append(values)
            except (ValueError, IndexError):
                continue


    if current_sample is not None and len(current_sample.get('data', [])) > 0:
        samples.append(current_sample)

    if not samples:
        raise ValueError(f"No valid samples found in file")


    if global_channel_names:
        channel_names = global_channel_names
    else:
        first_data = samples[0]['data']
        num_channels = len(first_data[0]) if first_data else 0
        channel_names = [f"Channel_{i+1}" for i in range(num_channels)]

    num_channels = len(channel_names)


    X_wave_list = []
    y_ft_list = []
    y_deg_list = []
    run_list = []

    for sample in samples:
        data = np.array(sample['data'])

        if len(data) == 0:
            continue


        if data.shape[1] != num_channels:
            if data.shape[1] < num_channels:
                pad_width = num_channels - data.shape[1]
                data = np.pad(data, ((0, 0), (0, pad_width)), mode='constant')
            else:
                data = data[:, :num_channels]


        if len(data) > max_len:
            indices = np.linspace(0, len(data)-1, max_len, dtype=int)
            data = data[indices]
        elif len(data) < max_len:
            pad_size = max_len - len(data)
            data = np.vstack([data, np.repeat(data[-1:], pad_size, axis=0)])

        X_wave_list.append(data)
        y_ft_list.append(sample['fault_type'])
        y_deg_list.append(sample['degradation'])
        run_list.append(sample['run'])

    X_wave = np.array(X_wave_list)
    y_ft = np.array(y_ft_list)
    y_deg = np.array(y_deg_list)
    runs = np.array(run_list)

    metadata = {
        'y_fault_type': y_ft,
        'y_degradation': y_deg,
        'runs': runs,
        'num_channels': num_channels,
        'channel_names': channel_names,
        'num_samples': len(X_wave),
        'sequence_length': max_len
    }

    if verbose:
        print(f"\n✓ Parsed {len(X_wave)} samples")
        print(f"  Shape: {X_wave.shape}")
        print(f"  Fault types: {sorted(np.unique(y_ft))}")
        print(f"  Degradation levels: {sorted(np.unique(y_deg))}")

    return X_wave, channel_names, metadata



class FlexibleOrdinalDataset(Dataset):
    def __init__(self, X_wave, X_stat, y_ft, y_deg_norm,
                 wave_mean=None, wave_std=None,
                 stat_mean=None, stat_std=None,
                 augment=False):

        self.X_wave_raw = X_wave
        self.X_stat_raw = X_stat
        self.y_ft = torch.tensor(y_ft, dtype=torch.long)

        self.y_deg_class = torch.tensor(
            np.round(y_deg_norm * 5).astype(np.int64).clip(0, 5),
            dtype=torch.long
        )
        self.y_deg_norm = torch.tensor(y_deg_norm, dtype=torch.float32)

        self.augment = augment

        if wave_mean is None:
            self.wave_mean = np.mean(X_wave, axis=(0, 1))
            self.wave_std = np.std(X_wave, axis=(0, 1)) + 1e-8
            self.stat_mean = np.mean(X_stat, axis=0)
            self.stat_std = np.std(X_stat, axis=0) + 1e-8
        else:
            self.wave_mean = wave_mean
            self.wave_std = wave_std
            self.stat_mean = stat_mean
            self.stat_std = stat_std

        self.X_wave = torch.tensor(
            (X_wave - self.wave_mean) / self.wave_std,
            dtype=torch.float32
        ).permute(0, 2, 1)

        self.X_stat = torch.tensor(
            (X_stat - self.stat_mean) / self.stat_std,
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.y_ft)

    def __getitem__(self, idx):
        x_wave = self.X_wave[idx]
        x_stat = self.X_stat[idx]

        if self.augment:
            x_wave = x_wave + torch.randn_like(x_wave) * 0.01
            x_stat = x_stat + torch.randn_like(x_stat) * 0.01

        return x_wave, x_stat, self.y_ft[idx], self.y_deg_class[idx]
