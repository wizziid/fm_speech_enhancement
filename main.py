import os
import torch
import torch.utils.data
import torchaudio
from dataset import GetDataset
from interpolant import Interpolant
from updated_network import Network
from sampler import StochasticSampler
import utils

CHECKPOINT = 0

def main():
    # Ensure necessary directories exist
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("artefacts/stft", exist_ok=True)
    os.makedirs("artefacts/wav", exist_ok=True)
    os.makedirs("artefacts/loss", exist_ok=True)
    
    # Set device preference
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    if device == "mps": torch.mps.empty_cache()
    if device == "cuda": torch.cuda.set_device(2)
    print(f"Using device: {device}")
    
    # Load dataset and dataloader
    dataset = GetDataset("data/", device=device)
    dataloader = dataset.get_dataloader(batch_size=1)
    dataset.print_info()
    
    # Initialize model, interpolant, and sampler
    vector_field_net = Network(input_shape=dataset.real_shape, device=device)
    interpolant = Interpolant()
    sampler = StochasticSampler(data_shape=dataset.real_shape, vector_field=vector_field_net, device=device)
    optimizer = torch.optim.Adam(vector_field_net.parameters(), lr=1e-4)
    total_params = sum(p.numel() for p in vector_field_net.parameters())
    utils.print_memory("Model initialised", device=device)
    print(f"Total Model Parameters: {total_params:,}")
    
    # Load model checkpoint if available
    model_path = "checkpoints/vector_field_new_net.pth"
    start_epoch = utils.load_model(vector_field_net, optimizer, model_path)
    losses = []
    
    # Training loop
    for epoch in range(start_epoch, start_epoch + 1):
        epoch_loss = 0.0
        for b, batch in enumerate(dataloader):
            optimizer.zero_grad()
            x1, x0 = batch
            x1, x0 = dataset.complex_to_real(x1.to(device)), dataset.complex_to_real(x0.to(device))
            t = utils.sample_time(batch_size=x0.shape[0]).to(device)
            It, time_derivative = interpolant.compute_interpolant(x0, x1, t)
            predicted_vector_field = vector_field_net(It, t)
            loss = torch.nn.functional.mse_loss(predicted_vector_field, time_derivative)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(f"epoch: {epoch} batch: {b}/{len(dataloader)} loss: {loss.item()}")
        
        # Log epoch loss and save checkpoint
        print(f"Epoch {epoch}: Loss = {epoch_loss / len(dataloader)}")
        losses.append(epoch_loss / len(dataloader))
        if epoch > 5: utils.plot_losses(losses, epoch)
        utils.save_model(vector_field_net, optimizer, (epoch + 1), model_path)
        
        # Evaluate model on test batch
        t, b = dataset.get_test_batch(batch_size=3)
        b, t = dataset.complex_to_real(b).to(device), dataset.complex_to_real(t).to(device)
        print(b.shape, t.shape)
        vector_field_net.eval()
        utils.save_sample(dataset, sampler, epoch, b, t)
        vector_field_net.train()

if __name__ == "__main__":
    main()