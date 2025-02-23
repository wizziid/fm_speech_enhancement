import torch
import utils

class Interpolant:
    def __init__(self, mode="trig"):
        """
        Initializes the interpolant.

        Modes:
        - "linear": Linear interpolation.
        - "trig": Trigonometric (sinusoidal) interpolation.
        """
        if mode not in ["linear", "trig"]:
            raise ValueError("Invalid mode. Choose 'linear' or 'trig'.")
        self.mode = mode

    def compute_interpolant(self, x0, x1, t):
        """
        Computes the interpolant and its time derivative.

        Args:
            x0 (torch.Tensor): Start point.
            x1 (torch.Tensor): End point.
            t (torch.Tensor): Time step in [0,1].

        Returns:
            tuple: (Interpolated value It, Time derivative)
        """
        t_expanded = utils.expand_time_like(t, x0)

        if self.mode == "linear":
            It = (1 - t_expanded) * x0 + t_expanded * x1
            time_derivative = x1 - x0

        elif self.mode == "trig":
            It = torch.sin((1 - t_expanded) * torch.pi / 2) * x0 + torch.sin(t_expanded * torch.pi / 2) * x1
            time_derivative = (torch.pi / 2) * (torch.cos(t_expanded * torch.pi / 2) * x1 - torch.cos((1 - t_expanded) * torch.pi / 2) * x0)

        return It, time_derivative
