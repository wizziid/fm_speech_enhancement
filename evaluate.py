import os
import torch
import torchaudio
import numpy as np
import scipy.signal
import noisereduce as nr
from dataset import GetDataset
from network import Network
from sampler import StochasticSampler
import utils
from pesq import pesq
from pystoi import stoi
from torchmetrics.audio import SignalDistortionRatio as SDR
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SNR_LEVELS = [0, 0.5, 1, 2]
NUM_SAMPLES = 3
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

dataset = GetDataset(root="data/", device=DEVICE)

def evaluate_model():
    print("Loading model...")
    model = Network(input_shape=dataset.real_shape, device=DEVICE)
    utils.load_model(model, None, "checkpoints/vector_field.pth")
    sampler = StochasticSampler(data_shape=dataset.real_shape, vector_field=model, device=DEVICE)
    model.eval()
    results_table = {}
    stft_storage = {}
    
    for snr in SNR_LEVELS:
        print(f"Processing SNR {snr} dB...")
        dataset.snr_db = snr
        os.makedirs(f"{RESULTS_DIR}/SNR_{snr}dB", exist_ok=True)
        metrics = {"model": {"pesq": [], "stoi": [], "sdr": []}, "wiener": {"pesq": [], "stoi": [], "sdr": []}, "reducenoise": {"pesq": [], "stoi": [], "sdr": []}}
        
        for sample_idx in range(NUM_SAMPLES):
            target, noisy = dataset.get_test_batch(batch_size=1)
            target, noisy = target.to(DEVICE), noisy.to(DEVICE)
            target_waveform = dataset.reconstruct_phase_istft(dataset.complex_to_real(target)).squeeze().cpu().numpy().astype(np.float32)
            noisy_waveform = dataset.reconstruct_phase_istft(dataset.complex_to_real(noisy)).squeeze().cpu().numpy().astype(np.float32)
            noisy_real = dataset.complex_to_real(noisy)
            enhanced_real = sampler.sample(noisy_real, iterations=10).detach().cpu()
            enhanced_waveform = dataset.reconstruct_phase_istft(enhanced_real[-1]).squeeze().cpu().numpy().astype(np.float32)
            wiener_denoised = scipy.signal.wiener(noisy_waveform).astype(np.float32)
            reducenoise_denoised = nr.reduce_noise(y=noisy_waveform, sr=dataset.sample_rate).astype(np.float32)
            
            for method, output in zip(["model", "wiener", "reducenoise"], [enhanced_waveform, wiener_denoised, reducenoise_denoised]):
                pesq_score = pesq(dataset.sample_rate, target_waveform, output, "wb")
                stoi_score = stoi(target_waveform, output, dataset.sample_rate, extended=False)
                sdr_score = SDR()(torch.tensor(output), torch.tensor(target_waveform)).item()
                metrics[method]["pesq"].append(pesq_score)
                metrics[method]["stoi"].append(stoi_score)
                metrics[method]["sdr"].append(sdr_score)
            
            save_sample(target_waveform, noisy_waveform, enhanced_waveform, wiener_denoised, reducenoise_denoised, snr, sample_idx)
            
            if sample_idx == 0:
                if snr not in stft_storage:
                    stft_storage[snr] = []
                stft_storage[snr].extend([dataset.stft(torch.tensor(noisy_waveform).unsqueeze(0)), dataset.stft(torch.tensor(target_waveform).unsqueeze(0)), dataset.stft(torch.tensor(enhanced_waveform).unsqueeze(0)), dataset.stft(torch.tensor(wiener_denoised).unsqueeze(0)), dataset.stft(torch.tensor(reducenoise_denoised).unsqueeze(0))])
        
        results_table[snr] = {method: {metric: np.mean(values) if values else None for metric, values in scores.items()} for method, scores in metrics.items()}
    
    save_results_table(results_table)
    plot_grid(stft_storage)

def save_sample(target, noisy, enhanced, wiener, reducenoise, snr, sample_idx):
    sample_dir = f"{RESULTS_DIR}/SNR_{snr}dB/sample_{sample_idx+1}"
    os.makedirs(sample_dir, exist_ok=True)
    torchaudio.save(f"{sample_dir}/waveform_original.wav", torch.tensor(noisy).unsqueeze(0), dataset.sample_rate)
    torchaudio.save(f"{sample_dir}/waveform_target.wav", torch.tensor(target).unsqueeze(0), dataset.sample_rate)
    torchaudio.save(f"{sample_dir}/waveform_enhanced.wav", torch.tensor(enhanced).unsqueeze(0), dataset.sample_rate)
    torchaudio.save(f"{sample_dir}/waveform_wiener.wav", torch.tensor(wiener).unsqueeze(0), dataset.sample_rate)
    torchaudio.save(f"{sample_dir}/waveform_reducenoise.wav", torch.tensor(reducenoise).unsqueeze(0), dataset.sample_rate)

def save_results_table(results):
    latex_path = os.path.join(RESULTS_DIR, "results_table.tex")
    with open(latex_path, "w") as f:
        f.write("\\begin{table}[h]\\centering\\begin{tabular}{|c|c|c|c|}\hline SNR (dB) & PESQ & STOI & SDR \\\hline")
        for snr, methods in results.items():
            f.write(f"{snr} & {methods['model']['pesq']:.3f} & {methods['model']['stoi']:.3f} & {methods['model']['sdr']:.3f} \\\hline")
        f.write("\\end{tabular}\\caption{Evaluation metrics for different SNR levels.}\\end{table}")

def plot_grid(stft_storage):
    fig, axes = plt.subplots(4, 5, figsize=(15, 10))
    snr_labels = ["0 dB", "0.5 dB", "1 dB", "2 dB"]
    method_labels = ["Model", "Wiener", "ReduceNoise", "Noisy", "Clean"]
    for row, snr in enumerate(SNR_LEVELS):
        for col, method_idx in enumerate(range(5)):
            axes[row, col].imshow(torch.abs(stft_storage[snr][method_idx]).log1p().squeeze().numpy(), aspect="auto", origin="lower", cmap="magma")
    plt.savefig(f"{RESULTS_DIR}/stft_comparison_grid.png")

if __name__ == "__main__":
    evaluate_model()
