import os
import torch
import psutil
import torchaudio
import matplotlib.pyplot as plt

"""
=======================================================================================================================
Samples, denormalises, converts back to complex spectrogram and wav form for saving.
=======================================================================================================================
"""


def save_sample(dataset, sampler, epoch, batch, target, iterations=100):
    batch_size = len(batch)
    sampled_spectrograms = sampler.sample(x0=batch, iterations=iterations, batch_size=batch_size).detach().cpu()
    time_indices = torch.linspace(0, iterations - 1, steps=4).long()
    fig, axes = plt.subplots(batch_size, 5, figsize=(15, 10))

    # Row = sample, column = time
    for col, t_idx in enumerate(time_indices):
        batch_spectrograms = sampled_spectrograms[t_idx]
        for row in range(batch_size):  
            spectrogram = dataset.real_to_complex(batch_spectrograms[row].unsqueeze(0))
            spectrogram = dataset.complex_denormalize(spectrogram)

            # Save an audio sample of the first sample denoised... Could probably be directly accessed with index.
            if row == 0 and col == 0:
                # output
                waveform = dataset.inverse_stft(spectrogram)
                audio_path = f"artefacts/wav/sample_audio_epoch_{epoch+1}_sample_{row}_out.wav"
                torchaudio.save(audio_path, waveform, dataset.sample_rate)
                # input
                input_waveform = dataset.inverse_stft(dataset.real_to_complex(batch[0].cpu().unsqueeze(0)))
                input_path = f"artefacts/wav/sample_audio_epoch_{epoch+1}_sample_{row}_input.wav"
                torchaudio.save(input_path, input_waveform, dataset.sample_rate)
                # Target
                target_waveform = dataset.inverse_stft(dataset.complex_denormalize(target[0]))
                target_path = f"artefacts/wav/sample_audio_epoch_{epoch+1}_sample_{row}_target.wav"
                torchaudio.save(target_path, target_waveform, dataset.sample_rate)

            # magnitudes
            img = torch.abs(spectrogram.squeeze()).log1p().numpy()
            axes[row, col].imshow(img, aspect="auto", origin="lower")
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

            # row/col labels
            if row == 0:
                axes[row, col].set_title(f"T={1 - col * 0.33:.2f}")  
            if col == 0:
                axes[row, col].set_ylabel(f"Sample {row}")

    # Plot target in finl column
    for row in range(batch_size):
        target_spectrogram = dataset.complex_denormalize(target[row].unsqueeze(0))  # Denormalize bc target is normalised in loader.
        img = torch.abs(target_spectrogram.squeeze()).log1p().numpy()
        axes[row, -1].imshow(img, aspect="auto", origin="lower")
        axes[row, -1].set_xticks([])
        axes[row, -1].set_yticks([])
        axes[row, -1].set_title("Target")  # Label target column

    plt.tight_layout()
    spectrogram_path = f"artefacts/stft/sample_spectrogram_epoch_{epoch+1}.png"
    plt.savefig(spectrogram_path)
    plt.close()
    print(f"Saved artefacts for epoch {epoch}")



def plot_losses(losses, epoch):
    save_dir = "artefacts/loss"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"loss_{epoch}.png")
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(losses) + 1), losses, marker='o', linestyle='-', color='b', label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Avg loss at epoch")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved loss curve to {save_path}")


def plot_stft(stft, title="STFT"):
    if stft.dim() == 3: 
        stft = stft[0]  
    plt.figure(figsize=(10, 5))
    plt.imshow(torch.abs(stft).log1p().numpy(), aspect='auto', origin='lower', cmap='magma')
    plt.colorbar(label="Magnitude (log scale)")
    plt.title(title)
    plt.xlabel("Time Frames")
    plt.ylabel("Frequency Bins")
    plt.show()


"""
=======================================================================================================================
    Terminal printouts
=======================================================================================================================
"""
def print_memory(stage, device="mps"):
    if device == "cuda" and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert bytes to GB
        print(f"[{stage}] Allocated CUDA Memory: {allocated:.2f} GB")

    elif device == "mps" and torch.backends.mps.is_available():
        allocated = torch.mps.current_allocated_memory() / (1024 ** 3)  # Convert bytes to GB
        print(f"[{stage}] Allocated MPS Memory: {allocated:.2f} GB")

    else:
        print(f"[{stage}] {device.upper()} not available.")

"""
=======================================================================================================================
    SAVING and LOADING model
=======================================================================================================================
"""
def save_model(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)

def load_model(model, optimizer, path):
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch']
    return 0

"""
=======================================================================================================================
    DIFFUSION helpers
=======================================================================================================================
"""

def sample_time(batch_size, device="cpu", min_val=0, max_val=1, use_lognormal=False):
    """
    Uniformally samples a time vector of shape (batch_size, 1) with values in [min_val, max_val].

    COULD TRY: Sample closer to sampling time like in EDM (they use log normal)
    """
    return min_val + (max_val - min_val) * torch.rand(batch_size, 1, device=device)

def fill_time(batch_size, value, device="cpu"):
    """
    Returns a time vector of shape (batch_size, 1) filled with a specific value.
    """
    return torch.full((batch_size, 1), value, device=device)

def expand_time_like(time_vector, data):
    """
    Expands a (batch_size, 1) time vector to match the shape of the data tensor.
    """
    return time_vector.view(data.shape[0], *([1] * (data.dim() - 1))).expand_as(data)


