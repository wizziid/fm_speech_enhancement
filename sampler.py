import torch
from torchdiffeq import odeint_adjoint as odeint
import utils

class StochasticSampler:
    def __init__(self, data_shape, vector_field, device="cpu"):
        self.data_shape = data_shape
        self.vector_field = vector_field.to(device)  
        self.device = torch.device(device)  
    
    def ode_rhs(self, t, x):
        x = x.to(self.device)  
        batch_size = x.shape[0] if x.dim() > 0 else 1  
        t_expanded = utils.fill_time(batch_size, t.item(), device=self.device) 
        return self.vector_field(x, t_expanded) if self.vector_field else torch.zeros_like(x, device=self.device)
    
    def sample(self, x0, iterations=10):
        time_grid = torch.linspace(0, 1, steps=iterations, device=self.device)  
        print(f"Starting ODE solver with x0 shape: {x0.shape}, device: {self.device}")
        x_samples = odeint(self.ode_rhs, x0, time_grid, method='rk4', atol=1e-8, rtol=1e-8, adjoint_params=(), options={"step_size": .1})
        print(f"ODE solver output shape: {x_samples.shape}, device: {x_samples.device}")
        return x_samples.to(self.device)
