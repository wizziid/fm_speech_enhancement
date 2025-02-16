import os

import matplotlib.pyplot as plt
import torch
import torchaudio
import numpy as np

"""
=======================================================================================================================
Samples, denormalises, converts back to complex spectrogram and wav form for saving.
=======================================================================================================================
"""

def save_sample(dataset, sampler, epoch, batch, target, iterations=10):
    batch_size = len(batch)
    sampled_spectrograms = sampler.sample(x0=batch, iterations=iterations, batch_size=batch_size).detach().cpu()
    time_indices = torch.round(torch.linspace(0, iterations - 1, steps=4)).long()
    print(time_indices)
    fig, axes = plt.subplots(batch_size, 5, figsize=(15, 10))

    print(f"Shape of Sample: {sampled_spectrograms.shape}")
    for s, spec in enumerate(sampled_spectrograms):
        print(f"Sample {s} min: {spec[0].min()}, max: {spec[0].max()}")

    print(f"target min: {dataset.complex_to_real(target[0].unsqueeze(0)).min()}, max: {dataset.complex_to_real(target[0].unsqueeze(0)).max()}")

    # Test
    
    # Get output of sampler and target and compare min max values at each step...
    sample_real = sampled_spectrograms[-1][0].unsqueeze(0)
    target_real = dataset.complex_to_real(target[0].unsqueeze(0))
    print(f"real values --- t_min: {target_real.min()}, t_max:{target_real.max()}")
    print(f"real values --- s_min: {sample_real.min()}, s_max:{sample_real.max()}")
    print()

    sample_complex = dataset.real_to_complex(sample_real)
    target_complex = dataset.real_to_complex(target_real)
    original_target_complex = target[0].unsqueeze(0)
    print(f"complex values --- t_min: {target_complex.detach().numpy().min()}, t_max:{target_complex.detach().numpy().max()}")
    print(f"complex values --- s_min: {sample_complex.detach().numpy().min()}, s_max:{sample_complex.detach().numpy().max()}")
    print(f"complex values --- o_t_min: {original_target_complex.detach().numpy().min()}, o_t_max: {original_target_complex.detach().numpy().min()}")
    print()

    sample_denorm = dataset.complex_denormalize(sample_complex)
    target_denorm = dataset.complex_denormalize(target_complex)
    original_target_denorm = dataset.complex_denormalize(original_target_complex)
    print(f"denormed --- t_min: {target_denorm.detach().numpy().min()}, t_max: {target_denorm.detach().numpy().max()}")
    print(f"denormed --- s_min: {sample_denorm.detach().numpy().min()}, s_max: {sample_denorm.detach().numpy().max()}")
    print(f"denormed --- o_t_min: {original_target_denorm.detach().numpy().min()}, o_t_max: {original_target_denorm.detach().numpy().max()}")
    print()


    same_indices = (target_denorm == original_target_denorm).nonzero(as_tuple=True)
    diff_indices = (target_denorm != original_target_denorm).nonzero(as_tuple=True)
    print(f"Same indices: {same_indices}")
    print(f"Different indices: {diff_indices}")

    

    sample_istft = dataset.inverse_stft(sample_denorm)
    target_istft = dataset.inverse_stft(target_denorm)
    original_target_istft = dataset.inverse_stft(original_target_denorm) 
    print(f"istft --- t_min: {target_istft.min()}, t_max: {target_istft.max()}")
    print(f"istft --- s_min: {sample_istft.min()}, t_max: {sample_istft.max()}")
    print(f"istft --- o_t_min: {original_target_istft.min()}, o_t_max: {original_target_istft.max()}")
    print()

    # Save waveforms
    output_waveform = dataset.inverse_stft(dataset.complex_denormalize(dataset.real_to_complex(sampled_spectrograms[-1][0].unsqueeze(0))))
    output_path = f"artefacts/wav/{epoch+1}_sample_0_out.wav"
    torchaudio.save(output_path, output_waveform, dataset.sample_rate)

    input_waveform = dataset.inverse_stft(dataset.complex_denormalize(dataset.real_to_complex(sampled_spectrograms[0][0].unsqueeze(0))))
    input_path = f"artefacts/wav/{epoch+1}_sample_0_input.wav"
    torchaudio.save(input_path, input_waveform, dataset.sample_rate)

    target_waveform = dataset.inverse_stft(dataset.complex_denormalize(target[0]))
    target_path = f"artefacts/wav/{epoch+1}_sample_0_target.wav"
    torchaudio.save(target_path, target_waveform, dataset.sample_rate)

    print(target_waveform.min(), target_waveform.max())
    print(output_waveform.min(), output_waveform.max())
    # Row = sample, column = time
    for col, t_idx in enumerate(time_indices):
        batch_spectrograms = sampled_spectrograms[t_idx]
        for row in range(batch_size):  
            spectrogram = dataset.real_to_complex(batch_spectrograms[row].unsqueeze(0))
            spectrogram = dataset.complex_denormalize(spectrogram)

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


