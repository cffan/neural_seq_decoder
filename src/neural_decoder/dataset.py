import numpy as np
import torch
from torch.utils.data import Dataset


class SpeechDataset(Dataset):
    def __init__(self, data, transform=None, num_threshold_crossings=128, num_spike_band_powers=128):
        self.data = data
        self.transform = transform
        self.n_days = len(data)
        self.n_trials = sum([len(d["sentenceDat"]) for d in data])

        self.neural_feats = []
        self.phone_seqs = []
        self.neural_time_bins = []
        self.phone_seq_lens = []
        self.days = []

        if (num_threshold_crossings == 128) & (num_spike_band_powers == 128):
            for day in range(self.n_days):
                for trial in range(len(data[day]["sentenceDat"])):
                    self.neural_feats.append(data[day]["sentenceDat"][trial])
                    self.phone_seqs.append(data[day]["phonemes"][trial])
                    self.neural_time_bins.append(data[day]["sentenceDat"][trial].shape[0])
                    self.phone_seq_lens.append(data[day]["phoneLens"][trial])
                    self.days.append(day)
        else:
            pcs_tc = np.load("../data/threshold_crossing_principal_components.npy")
            pcs_sbp = np.load("../data/spike_band_power_principal_components.npy")

            tc_mean = np.load("../data/threshold_crossings_mean.npy")
            sbp_mean = np.load("../data/spike_band_power_mean.npy")

            for day in range(self.n_days):
                for trial in range(len(data[day]["sentenceDat"])):
                    trial_data = data[day]["sentenceDat"][trial]
                    sample_tcs = trial_data.T[:128]
                    sample_sbp = trial_data.T[128:]

                    tc_new = (pcs_tc[:, :num_threshold_crossings].T @ (sample_tcs - tc_mean)).T
                    sbp_new = (pcs_sbp[:, :num_spike_band_powers].T @ (sample_sbp - sbp_mean)).T

                    trial_data_new = np.concatenate([tc_new, sbp_new], axis=1)

                    self.neural_feats.append(trial_data_new)
                    self.phone_seqs.append(data[day]["phonemes"][trial])
                    self.neural_time_bins.append(data[day]["sentenceDat"][trial].shape[0])
                    self.phone_seq_lens.append(data[day]["phoneLens"][trial])
                    self.days.append(day)

    def __len__(self):
        return self.n_trials

    def __getitem__(self, idx):
        neural_feats = torch.tensor(self.neural_feats[idx], dtype=torch.float32)

        if self.transform:
            neural_feats = self.transform(neural_feats)

        return (
            neural_feats,
            torch.tensor(self.phone_seqs[idx], dtype=torch.int32),
            torch.tensor(self.neural_time_bins[idx], dtype=torch.int32),
            torch.tensor(self.phone_seq_lens[idx], dtype=torch.int32),
            torch.tensor(self.days[idx], dtype=torch.int64),
        )
