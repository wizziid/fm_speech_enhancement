import os
import random
import matplotlib.pyplot as plt
import sounddevice as sd
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import DataLoader, Subset
from torchaudio.datasets import LIBRISPEECH
import torchaudio.functional as F
from tqdm import tqdm

class GetDataset:
    """
    Handles dataset loading, STFT transformations, noise addition, and data preprocessing for speech enhancement.
    """
    def __init__(self, root="data/", url="train-clean-100", sample_rate=16000, n_fft=254, hop_length=32, win_length=128, max_length_seconds=1, device="cpu"):
        os.makedirs(root, exist_ok=True)
        self.dataset = LIBRISPEECH(root=root, url=url, download=True)
        self.device = device
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.max_length = int(max_length_seconds * sample_rate) + (12 * hop_length) - 1
        self.alpha = 0.5
        self.beta = 1
        self.snr_db = 1
        self.max_length_seconds = max_length_seconds
        self.stft = T.Spectrogram(n_fft=n_fft, hop_length=hop_length, win_length=win_length, power=None, onesided=True)
        self.istft = T.InverseSpectrogram(n_fft=n_fft, hop_length=hop_length, win_length=win_length, onesided=True)
        self.example_spec, _ = self[10]
        self.complex_shape = (1, *self.example_spec.shape[1:])
        self.real_shape = (2, *self.example_spec.shape[1:])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        waveform, sr, _, _, _, _ = self.dataset[idx]
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        if waveform.shape[1] < self.max_length:
            pad = self.max_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        else:
            max_start = waveform.shape[1] - int(self.max_length)
            start = torch.randint(0, max_start + 1, (1,)).item()
            waveform = waveform[:, start : start + self.max_length]
        
        noise = torch.randn_like(waveform) * (torch.std(waveform) / (self.snr_db + 1e-10))
        signal_power = torch.mean(waveform ** 2)
        noise_power = torch.mean(noise ** 2)
        snr_linear = 10 ** (self.snr_db / 10)
        eps = 1e-10
        noise = noise * torch.sqrt(signal_power / (snr_linear * (noise_power + eps)))
        noised_waveform = waveform + noise
        spectrogram = self.complex_normalize(self.stft(waveform))
        noised_spectrogram = self.complex_normalize(self.stft(noised_waveform))
        return spectrogram, noised_spectrogram

    def get_dataloader(self, batch_size=16, shuffle=True):
        #clamped = Subset(self, range(2))
        #return DataLoader(clamped, batch_size=batch_size, shuffle=shuffle)
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)

    def complex_to_real(self, tensor):
        assert tensor.dim() == 4
        real_tensor = torch.view_as_real(tensor).squeeze(1)
        return real_tensor.permute(0, 3, 1, 2)

    def real_to_complex(self, tensor):
        assert tensor.dim() == 4
        tensor = tensor.permute(0, 2, 3, 1).contiguous()
        return torch.view_as_complex(tensor)

    def complex_normalize(self, tensor):
        magnitude, phase = torch.abs(tensor), torch.angle(tensor)
        return torch.polar(self.beta * magnitude ** self.alpha, phase)
    
    def complex_denormalize(self, tensor):
        magnitude, phase = torch.abs(tensor), torch.angle(tensor)
        return torch.polar((magnitude / self.beta) ** (1 / self.alpha), phase)

    def reconstruct_phase_istft(self, sampled_spectrogram):
        return self.istft(self.complex_denormalize(self.real_to_complex(sampled_spectrogram)))


    def reconstruct_phase_griffinlim(self, sampled_spectrogram):
        """
        Uses Griffin-Lim algorithm to reconstruct phase from magnitude.
        """
        sampled_complex = self.real_to_complex(sampled_spectrogram)
        sampled_complex = self.complex_denormalize(sampled_complex)
        
        magnitude = torch.abs(sampled_complex)
        reconstructed_waveform = F.griffinlim(
            magnitude,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            window=torch.hann_window(self.win_length, device=magnitude.device),
            power=1.0,
            n_iter=64,  
            momentum=0.99, 
            length=None,  
            rand_init=True 
        )
        return reconstructed_waveform

    def reconstruct_phase_noisy(self, sampled_spectrogram, noisy_spectrogram, threshold=.001):
        """
        Uses noisy input's phase for bins where the predicted magnitude is above a threshold.
        """
        sampled_complex = self.real_to_complex(sampled_spectrogram)
        sampled_complex = self.complex_denormalize(sampled_complex)
        noisy_complex = self.real_to_complex(noisy_spectrogram)
        noisy_complex = self.complex_denormalize(noisy_complex)
        sampled_magnitude = torch.abs(sampled_complex)
        noisy_phase = torch.angle(noisy_complex)
        reconstructed_complex = torch.polar(sampled_magnitude, noisy_phase)
        return self.istft(reconstructed_complex)




    def get_test_batch(self, batch_size):
        length = self.sample_rate * 2 + (self.hop_length * 23)
        inputs, targets = [], []
        for i in [random.randint(0, len(self.dataset)) for _ in range(batch_size)]:
            waveform, sr, _, _, _, _ = self.dataset[i]
            if waveform.shape[1] < length:
                waveform = torch.nn.functional.pad(waveform, (0, length - waveform.shape[1]))
            else:
                max_start = waveform.shape[1] - int(length)
                start = torch.randint(0, max_start + 1, (1,)).item()
                waveform = waveform[:, start : start + length]
            noise = torch.randn_like(waveform) * (torch.std(waveform) / (self.snr_db + 1e-10))
            signal_power = torch.mean(waveform ** 2)
            noise_power = torch.mean(noise ** 2)
            snr_linear = 10 ** (self.snr_db / 10)
            eps = 1e-10
            noise = noise * torch.sqrt(signal_power / (snr_linear * (noise_power + eps)))
            noised_waveform = waveform + noise
            spectrogram = self.complex_normalize(self.stft(waveform))
            noised_spectrogram = self.complex_normalize(self.stft(noised_waveform))
            inputs.append(noised_spectrogram)
            targets.append(spectrogram)
        return torch.stack(targets), torch.stack(inputs)

    def print_info(self):
        out = (
            "\nSTFT parameters:\n"
            f"\tSample Rate: {self.sample_rate}\n"
            f"\tn_fft: {self.n_fft}\n"
            f"\tWindow: {self.win_length}\n"
            f"\tHop Length: {self.hop_length}\n\n"
            "Data:\n"
            f"\tComplex shape: {self.complex_shape}\n"
            f"\tReal shape: {self.real_shape}\n"
            f"\tNumber of datapoints: {len(self)}\n"
            f"\tDatapoint length (seconds): {self.max_length_seconds}\n"
            f"\tSample max: {self.example_spec.detach().numpy().max()}\n"
            f"\tSample min: {self.example_spec.detach().numpy().min()}\n"
        )
        print(out)
