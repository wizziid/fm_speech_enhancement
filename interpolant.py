import torch
import utils

class Interpolant:
    def __init__(self, mode="trig"):
        """
        Initializes the interpolant.

        Can use trig or linear
        """
        self.mode = mode
    
    def compute_interpolant(self, x0, x1, t):
        """
        Computes the interpolant and its time derivative based on the selected mode.
        """
        t_expanded = utils.expand_time_like(t, x0)
        
        if self.mode == "linear":
            It = (1 - t_expanded) * x0 + t_expanded * x1 
            time_derivative = x1 - x0
        
        elif self.mode == "trig":
            It = torch.sin((1 - t_expanded) * torch.pi / 2) * x0 + torch.sin(t_expanded * torch.pi / 2) * x1
            time_derivative = (torch.pi / 2) * (torch.cos(t_expanded * torch.pi / 2) * x1 - torch.cos((1 - t_expanded) * torch.pi / 2) * x0)
        else:
            raise ValueError("Invalid mode. Choose 'linear', 'trig', or 'vp'.")
        
        return It, time_derivative

