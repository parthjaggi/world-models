
"""
Variational encoder model, used as a visual model
for our model of the world.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.vae_models import Encoder, Decoder, Encoder_v2, Decoder_v2

class VAE(nn.Module):
    """ Variational Autoencoder """
    def __init__(self, img_channels, latent_size, args):
        super(VAE, self).__init__()
        self.encoder = Encoder(img_channels, latent_size, args)
        self.decoder = Decoder(img_channels, latent_size, args)

    def forward(self, x): # pylint: disable=arguments-differ
        mu, logsigma = self.encoder(x)
        sigma = logsigma.exp()
        eps = torch.randn_like(sigma)
        z = eps.mul(sigma).add_(mu)

        recon_x = self.decoder(z)
        return recon_x, mu, logsigma
