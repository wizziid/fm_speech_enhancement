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
    For STFT, resulting number of frequencies is n_fft/2 + 1. so to get 64 frequency bins, need n_fft (64-1) * 2.
    Also win_length can't be more than this.
    Also hop_length can't be more than win_length.
    
    Have hardcoded sample rate to specifically output time bins that are powers of 2...
    """

    def __init__(self, root="data/", url="train-clean-100", sample_rate=16000, n_fft=254, hop_length=64, win_length=128, max_length_seconds=1, device="cpu"):
        os.makedirs(root, exist_ok=True)
        # make more data
        self.dataset = LIBRISPEECH(root=root, url=url, download=True)
        self.device = device
        # STFT PARAMS
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        # hard code sample length so that stft has power of 2 time bins... 
        self.max_length = max_length_seconds * sample_rate + (6 * hop_length) - 1 
        self.max_length_seconds = max_length_seconds
        # NORMALISATION PARAMS
        self.alpha = 0.5
        self.beta = 1  # 0.5 
        # NOISE PARAMS
        self.snr_db = 1 

        self.stft = T.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            power=None,
            onesided= True  #False  
        )

        self.istft = T.InverseSpectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            onesided= True  #False
        )

        self.example_spec, _, _ = self[10]
        self.complex_shape = (1, *self.example_spec.shape[1:])  # Complex (Batch, 1, Frequency, Time)
        self.real_shape = (2, *self.example_spec.shape[1:])  # Real (Batch, 2, Frequency, Time)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # -> (waveform, sample_rate: int, transcript, speaker_id: int, chapter_id: int, utterance_id: int).
        waveform, sr, _, _, _, _  = self.dataset[idx]
        # sample rate check
        
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Pad or truncate to sample_rate * seconds.
        if waveform.shape[1] < self.max_length:
            pad = self.max_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        else:
            # Randomly sample a starting point
            max_start = waveform.shape[1] - int(self.max_length)
            start = torch.randint(0, max_start + 1, (1,)).item()
            waveform = waveform[:, start : start + self.max_length]

        # Add Noise    
        
	# use random snr_db ratio??
	# snr_db = 0.5 + torch.random((1,)) * 5  # min + [0, 1] * (max - min)
	noise = torch.randn_like(waveform) * (torch.std(waveform) / (self.snr_db + 1e-10))
        signal_power = torch.mean(waveform ** 2)
        noise_power = torch.mean(noise ** 2)
        # print(f"signal power {signal_power}, noise power {noise_power}, mean wav {waveform.mean()}, wav min {waveform.min()}, wav max {waveform.max()}")
        snr_linear = 10 ** (self.snr_db / 10)
        eps = 1e-10
        noise = noise * torch.sqrt(signal_power / (snr_linear * (noise_power + eps)))
        noised_waveform = waveform + noise
        mask = (noise != 0).float()

        #print(f"s_power {signal_power}, n_power {noise_power}")
        #print(f"wav min {waveform.min()}, wav max {waveform.max()}")
        #print(f"noise min {noise.min()}, noise max {noise.max()}")
        #print(f"Noised wav min {noised_waveform.min()}, noised wav max {noised_waveform.max()}")   
        #print() 
        #torchaudio.save("artefacts/noised.wav", noised_waveform, sample_rate=16000)
        #torchaudio.save("artefacts/original.wav", waveform, sample_rate=16000)

        # convert original, noised and mask into normalised complex stft
        spectrogram = self.stft(waveform)
        spectrogram = self.complex_normalize(spectrogram)
        noised_spectrogram = self.stft(noised_waveform)
        noised_spectrogram = self.complex_normalize(noised_spectrogram)
        mask_spectrogram = self.stft(mask)
        mask_spectrogram = self.complex_normalize(mask_spectrogram)

        return spectrogram, noised_spectrogram, mask_spectrogram

    def get_dataloader(self, batch_size=32, shuffle=True):
        # clamped = Subset(self, range(256))
        # return DataLoader(clamped, batch_size=batch_size, shuffle=shuffle)
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)

    def plot_spectrogram(self, spectrogram, title="Spectrogram"):
        plt.figure(figsize=(10, 5))
        plt.imshow(torch.abs(spectrogram).log1p().squeeze().numpy(), aspect='auto', origin='lower')
        plt.colorbar()
        plt.title(title)
        plt.show()
    
    def inverse_stft(self, spectrogram):
        return self.istft(spectrogram)
    
    def play_audio(self, waveform):
        """
        Plays the waveform on sound device with the given sample rate.
        """
        sd.play(waveform.numpy().T, samplerate=self.sample_rate)
        sd.wait()

    def complex_to_real(self, tensor):
        """
        Transform complex tensor into real values 2 channel tensor. More compatible with torch modules.
        """
        assert tensor.dim()==4, f"Tensor should have 4 dimensions, received {tensor.dim()}"
        real_tensor = torch.view_as_real(tensor)  # Get real number representation
        real_tensor = real_tensor.squeeze(1)  # Torch adds extra dim through view_as_real for some reason
        real_tensor = real_tensor.permute(0, 3, 1, 2)  # Permute to (batch, 2, freq, time) 

        return real_tensor

    def real_to_complex(self, tensor):
        """
        Transforms real tensor into complex tensor.
        """
        assert tensor.dim() == 4
        tensor = tensor.permute(0, 2, 3, 1).contiguous()  # Permute to (batch, freq, time, 2) 
        complex_tensor = torch.view_as_complex(tensor)  # Convert back to complex representation
        return complex_tensor


    def is_real_mode(self, tensor):
        """
        Check for use in assertions
        """
        return tensor.dim() >= 4 and tensor.shape[1] == 2  


    def complex_normalize(self, tensor):
        """
        Follows normalisation process from https://arxiv.org/pdf/2208.05830 - section 2.a
        """
        magnitude = torch.abs(tensor)
        phase = torch.angle(tensor)
        new_magnitude = self.beta * magnitude ** self.alpha
        normalized_tensor = torch.polar(new_magnitude, phase)
        return normalized_tensor

    def complex_denormalize(self, tensor):
        """
        denormalise above normalisation
        """
        magnitude = torch.abs(tensor)
        phase = torch.angle(tensor)
        original_magnitude = (magnitude / self.beta) ** (1 / self.alpha)
        denormalized_tensor = torch.polar(original_magnitude, phase)
        return denormalized_tensor
    
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

    def reconstruct_phase_istft(self, sampled_spectrogram):
        """
        Uses inverse STFT to reconstruct waveform (baseline method).
        """
        sampled_complex = self.real_to_complex(sampled_spectrogram)
        sampled_complex = self.complex_denormalize(sampled_complex)
        return self.istft(sampled_complex)


    def reconstruct_phase_griffinlim(self, sampled_spectrogram):
        """
        Uses Griffin-Lim algorithm to reconstruct phase from magnitude.
        """
        sampled_complex = self.real_to_complex(sampled_spectrogram)
        sampled_complex = self.complex_denormalize(sampled_complex)
        
        magnitude = torch.abs(sampled_complex)
        #power = magnitude**2

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

    def reconstruct_phase_threshold(self, sampled_spectrogram, noisy_spectrogram, threshold=.001):
        """
        Uses noisy input's phase for bins where the predicted magnitude is above a threshold.
        """
        sampled_complex = self.real_to_complex(sampled_spectrogram)
        #print(sampled_complex.device)
        sampled_complex = self.complex_denormalize(sampled_complex)
        noisy_complex = self.real_to_complex(noisy_spectrogram)
        # print(noisy_complex.device)
        noisy_complex = self.complex_denormalize(noisy_complex)
        # Compute magnitudes
        sampled_magnitude = torch.abs(sampled_complex)
        noisy_phase = torch.angle(noisy_complex)
        # Create mask based on threshold
        mask = sampled_magnitude > threshold
        # Combine sampled magnitude with noisy input phase where mask is True
        # reconstructed_complex = torch.polar(sampled_magnitude, noisy_phase * mask)

        reconstructed_complex = torch.polar(sampled_magnitude, noisy_phase) #* mask)

        return self.istft(reconstructed_complex)

