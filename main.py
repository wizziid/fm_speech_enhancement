
import os

import torch
import torch.utils.data
import torchaudio

from dataset import GetDataset
from interpolant import Interpolant
from network import Network
from sampler import StochasticSampler
import utils


CHECKPOINT = 0

def main():
    torch.manual_seed(589)
    # ensure directories exist
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("artefacts", exist_ok=True)
    os.makedirs("artefacts/stft", exist_ok=True)
    os.makedirs("artefacts/wav", exist_ok=True)
    os.makedirs("artefacts/loss", exist_ok=True)
    # Device handling
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        device = "cuda"
        torch.cuda.set_device(2)
    print(f"Using device: {device}")

    # Get dataset and dataloader
    dataset = GetDataset("data/", device=device)
    dataloader = dataset.get_dataloader(batch_size=32)
    dataset.print_info()

    # Initialize network, interpolant, and sampler
    vector_field_net = Network(input_shape=dataset.real_shape, device=device)
    interpolant = Interpolant()
    sampler = StochasticSampler(data_shape=dataset.real_shape, vector_field=vector_field_net, device=device)
    optimizer = torch.optim.Adam(list(vector_field_net.parameters()), lr=1e-5)
    total_params = sum(p.numel() for p in vector_field_net.parameters())
    utils.print_memory("Model initialised", device=device)
    print(f"Total Model Parameters: {total_params:,}")

    # Load previous checkpoint if exists
    model_path = f"checkpoints/vector_field_126_freq.pth"
    start_epoch = utils.load_model(vector_field_net, optimizer, model_path)
    losses = []

    # Training loop
    for epoch in range(start_epoch, start_epoch + 100):
        epoch_loss = 0.0
        for b, batch in enumerate(dataloader):
            optimizer.zero_grad()  # zero grad
            x1, x0, mask = batch
            x1 = x1.to(device)
            x1 = dataset.complex_to_real(x1)
            x0 = x0.to(device)
            x0 = dataset.complex_to_real(x0)
            t = utils.sample_time(batch_size=x0.shape[0]).to(device)
            It, time_derivative = interpolant.compute_interpolant(x0, x1, t)  # Compute interpolant, time derivative
            # utils.print_memory("Interpolant and time derivative Loaded")
            predicted_vector_field = vector_field_net(It, t)  # predict vector from neural network
            # utils.print_memory("Forward pass")
            loss = torch.nn.functional.mse_loss(predicted_vector_field, time_derivative) 
            loss.backward()
            # utils.print_memory("Gradients calculated")
            optimizer.step()
            epoch_loss += loss.item()
            print(f"epoch: {epoch} batch: {b}/{len(dataloader)} loss: {loss.item()}")
        
        print(f"Epoch {epoch}: Loss = {epoch_loss / len(dataloader)}")
        losses.append(epoch_loss/len(dataloader))

        if epoch > 5:
            utils.plot_losses(losses, epoch)
    
        # enhance a couple of examples from the dataloader
        for b, batch in enumerate(dataloader):
            t, b, _ = batch
            t = t[:3]
            b = b[:3].to(device)
            b = dataset.complex_to_real(b)
            vector_field_net.eval()
            utils.save_sample(dataset, sampler, epoch, b, t)
            vector_field_net.train()
            break

        utils.save_model(vector_field_net, optimizer, (epoch + 1), model_path)

if __name__ == "__main__":
    main()
