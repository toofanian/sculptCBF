from cbf_opt.cbf import ControlAffineCBF
import numpy as np
import torch


class NeuralControlAffineCBF(ControlAffineCBF):
    def __init__(self, dynamics, params, **kwargs):
        self.dynamics = dynamics
        self.V_nn = kwargs.get("V_nn")
        self.normalizer = kwargs.get("normalizer")
        self.normalize_gradient_w_radius = kwargs.get("normalize_gradient_w_radius", False)
        super().__init__(dynamics, params, test=False, **kwargs)

    def vf(self, x: np.ndarray, time: float = 0.0) -> np.ndarray:
        x_norm = self.normalizer.standardize(x)
        return -self.V_nn(torch.Tensor(x_norm)).detach().numpy().squeeze()

    def _grad_vf(self, x: np.ndarray, time: float = 0.0) -> np.ndarray:
        x_norm = self.normalizer.standardize(x)
        x_norm_tensor = torch.Tensor(x_norm)
        x_norm_tensor.requires_grad = True

        if x_norm_tensor.ndim == 2:
            grad_vfs = []
            for x_norm_elem in x_norm_tensor:
                cbf_val = self.V_nn(x_norm_elem)
                grad_vf = torch.autograd.grad(cbf_val, x_norm_elem)[0].detach().numpy()
                grad_vfs.append(grad_vf)
            grad_vfs = np.array(grad_vfs)
        else:
            cbf_val = self.V_nn(x_norm_tensor)
            grad_vfs = torch.autograd.grad(cbf_val, x_norm_tensor)[0].detach().numpy()
        return -grad_vfs
