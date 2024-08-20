import torch
from torch import nn , Tensor
from torch.nn import functional as F
import torchvision


class vae(nn.Module):


    def __init__(self, trained_weights: bool , latent_dim= None ) :
        super(vae, self).__init__()

        '''
        latent_dim : num of latent dimensions
        trained_weights: True or False
        '''
        if latent_dim is None:
            self.latent_dim = 256
        
        ftr_extractor = torchvision.models.resnet50(pretrained = trained_weights)
        ftr_extractor.fc = nn.Identity()

        self.encoder = ftr_extractor   #outftrs = 2048 
        self.fc_mu = nn.Linear(2048, self.latent_dim)
        self.fc_var = nn.Linear(2048, self.latent_dim)

        decode_dims = [3 , 16, 32, 64, 128, 256 , 512 , 1024 , 2048]
        modules = []
        self.decoder_input = nn.Linear(self.latent_dim, 2048)
        for i in range(len(decode_dims) - 1):
            modules.append(
                nn.Sequential(nn.ConvTranspose2d(decode_dims[- i-1],
                decode_dims[-i -2],kernel_size = 1 , stride =1)))
        self.decoder = nn.Sequential(*modules)

    def encode(self, input: Tensor ) -> list[torch.Tensor]:
        result = self.encoder(input)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> torch.Tensor:
        std =  torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) 
        eps =  torch.clamp(eps , 0 , 1)
        return eps * std + mu


    def decode(self, z: Tensor) -> torch.Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, 2048, 1, 1)
        result = self.decoder(result)
        return result

    def forward(self, input: Tensor, **kwargs) -> list[torch.Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        out = self.decode(z)
        _,_,h_out,w_out = out.size()
        _,_,h_in,w_in = input.size()
        if h_out != h_in or w_out != w_in:
            out = F.interpolate(out , [h_in , w_in] , mode = "bilinear" ,  align_corners =  True )
        else:
            pass
        
        return  [out, input, mu, log_var]



    def generate(self, x: Tensor, **kwargs) -> torch.Tensor:
        return self.forward(x)[0]


