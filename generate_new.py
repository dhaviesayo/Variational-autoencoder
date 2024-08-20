
import torch
from torch import nn , Tensor
from torch.nn import functional as F
from .variational_autoencoder import vae


def generate_new(self, img : Tensor ,  num_samples:int) -> list[Tensor] :

        mu, log_var = vae.encode(img)
        sample_params = [vae.reparameterize(mu , log_var) for i in range (num_samples)]
        decoded = [vae.decode(params) for params in sample_params]
        decoded_interp = [F.interpolate(decoded[i] , [img.shape[-2] , 
        img.shape[-1]] , mode = "bilinear" ,  align_corners =  True ) for i in range(decoded)]
        return  decoded_interp