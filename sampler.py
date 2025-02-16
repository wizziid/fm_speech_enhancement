import torch

from torchdiffeq import odeint_adjoint as odeint

import utils


class StochasticSampler:

    def __init__(self, data_shape, vector_field, device="cpu"):
        """
        Initializes the sampler with data shape and neural network for the vector field.
        """
        self.data_shape = data_shape
        self.vector_field = vector_field.to(device)  
        self.device = torch.device(device)  
    
    def ode_rhs(self, t, x):
        """
        Defines the ODE right-hand side (dx/dt) based on the learned vector field.
        """
        x = x.to(self.device)  
        batch_size = x.shape[0] if x.dim() > 0 else 1  
        t_expanded = utils.fill_time(batch_size, t.item(), device=self.device) 
        v_field = self.vector_field(x, t_expanded) if self.vector_field else torch.zeros_like(x, device=self.device)
        # print(f"t: {t}, x shape: {x.shape}, v_field shape: {v_field.shape}, device: {x.device}")
        return v_field 
    
    def sample(self, x0, iterations=10, batch_size= 1):
        """
        Samples new data points using the probability flow ODE solver.
        """
        time_grid = torch.linspace(0, 1, steps=iterations, device=self.device)  
        print(f"Starting ODE solver with x0 shape: {x0.shape}, device: {self.device}")
        x_samples = odeint(
            self.ode_rhs, 
            x0, 
            time_grid, 
            method='rk4', 
            atol=1e-8,
            rtol=1e-8,
            adjoint_params=(),
            options={"step_size":.1}  #, "dtype":torch.float32}
        )  # Solve ODE
        print(f"ODE solver output shape: {x_samples.shape}, device: {x_samples.device}")
        return x_samples.to(self.device)  
