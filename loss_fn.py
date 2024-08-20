import torch
from torch import nn , Tensor
from torch.nn import functional as F


def loss_function(self,params: list , kld_weight) -> dict :
        '''
        params : list of output from the forward pass [reconstructed image ,  input , mu , log_var]
        kld_weight # Account for the minibatch samples from the dataset
        '''
        recons ,  input , mu , log_var = params
        def ssim (x,y):
            C1 = 0.01 ** 2
            C2 = 0.03 ** 2

            mu_x = nn.AvgPool2d(3, 1)(x)
            mu_y = nn.AvgPool2d(3, 1)(y)
            mu_x_mu_y = mu_x * mu_y
            mu_x_sq = mu_x.pow(2)
            mu_y_sq = mu_y.pow(2)

            sigma_x = nn.AvgPool2d(3, 1)(x * x) - mu_x_sq
            sigma_y = nn.AvgPool2d(3, 1)(y * y) - mu_y_sq
            sigma_xy = nn.AvgPool2d(3, 1)(x * y) - mu_x_mu_y

            SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
            SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
            SSIM = SSIM_n / SSIM_d

            return torch.clamp((1 - SSIM) / 2, 0, 1)

        def loss_fn(x,y , ssim = ssim):
            l1 = torch.mean(torch.abs(x - y))

            ssim = torch.mean(ssim(x , y))

            image_loss= 0.85 * ssim + (1 - 0.85) * l1

            return image_loss

        recons_loss = loss_fn(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        loss = recons_loss + (kld_weight * kld_loss)
        ## only the loss key in the dictionary would be used in the backpropagation
        
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}
