import torch

import numpy as np



def add_gaussian_noise(img_tensor, sigma=0.1):

    noise = torch.randn_like(img_tensor) * sigma
    noisy = torch.clamp(img_tensor + noise, 0, 1)

    return noisy

