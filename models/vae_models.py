
"""
Variational encoder model, used as a visual model
for our model of the world.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

enc_out = 2048

# V1: latent 32.
# V2: latent 200, dtse compatible enc, dec models. 
    # lowest loss: ~28-32
# V3: latent 200, kernel 200, single conv and deconv layer (relu)
    # lowest loss: 150 epochs. both train and test loss at 27.xx
# V3.2: latent 200, kernel 200, single conv (32->128) and deconv layer (sigmoid, 32->128)
    # lowest loss: complete training. 26.7 to 26.9
    # updated to run both test4_2 and test0_1

class Encoder_v2(nn.Module):
    """ VAE encoder """
    def __init__(self, img_channels, latent_size):
        super(Encoder, self).__init__()
        self.latent_size = latent_size
        #self.img_size = img_size
        self.img_channels = img_channels

        self.conv1 = nn.Conv2d(img_channels, 32, [1, 16], stride=[1, 2])
        self.conv2 = nn.Conv2d(32, 64, [1, 16], stride=[1, 2])
        self.conv3 = nn.Conv2d(64, 128, [1, 16], stride=[1, 2])
        self.conv4 = nn.Conv2d(128, 128, [1, 8], stride=[1, 2])

        # self.fc_mu = nn.Linear(2*2*256, latent_size)
        # self.fc_logsigma = nn.Linear(2*2*256, latent_size)

        self.fc_mu = nn.Linear(6144, latent_size)
        self.fc_logsigma = nn.Linear(6144, latent_size)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)

        mu = self.fc_mu(x)
        logsigma = self.fc_logsigma(x)

        return mu, logsigma

class Decoder_v2(nn.Module):
    """ VAE decoder """
    def __init__(self, img_channels, latent_size):
        super(Decoder, self).__init__()
        self.latent_size = latent_size
        self.img_channels = img_channels

        # self.fc1 = nn.Linear(latent_size, 1024)
        # self.deconv1 = nn.ConvTranspose2d(1024, 128, 5, stride=2)
        # self.deconv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        # self.deconv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        # self.deconv4 = nn.ConvTranspose2d(32, img_channels, 6, stride=2)

        self.fc1 = nn.Linear(latent_size, 6144)
        self.deconv1 = nn.ConvTranspose2d(128, 128, [1, 8], stride=[1, 2])
        self.deconv2 = nn.ConvTranspose2d(128, 64, [1, 16], stride=[1, 2])
        self.deconv3 = nn.ConvTranspose2d(64, 32, [1, 16], stride=[1, 2])
        self.deconv4 = nn.ConvTranspose2d(32, img_channels, [1, 22], stride=[1, 2])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = x.unsqueeze(-1).unsqueeze(-1)
        x = x.view(x.size(0), -1, 16, 3)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        reconstruction = torch.sigmoid(self.deconv4(x))
        return reconstruction

class Decoder(nn.Module):
    """ VAE decoder """
    def __init__(self, img_channels, latent_size, args):
        super(Decoder, self).__init__()
        self.latent_size = latent_size
        self.img_channels = img_channels

        # self.fc1 = nn.Linear(latent_size, 1024)
        # self.deconv1 = nn.ConvTranspose2d(1024, 128, 5, stride=2)
        # self.deconv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        # self.deconv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        # self.deconv4 = nn.ConvTranspose2d(32, img_channels, 6, stride=2)

        self.fc1 = nn.Linear(latent_size, enc_out)
        self.deconv1 = nn.ConvTranspose2d(128, 1, [1, 200], stride=1)
        # self.deconv2 = nn.ConvTranspose2d(128, 64, [1, 16], stride=[1, 2])
        # self.deconv3 = nn.ConvTranspose2d(64, 32, [1, 16], stride=[1, 2])
        # self.deconv4 = nn.ConvTranspose2d(32, img_channels, [1, 22], stride=[1, 2])

    def forward(self, x): # pylint: disable=arguments-differ
        x = F.relu(self.fc1(x))
        # x = x.unsqueeze(-1).unsqueeze(-1)
        x = x.view(x.size(0), 128, -1, 1)
        x = F.relu(self.deconv1(x))
        # x = F.relu(self.deconv2(x))
        # x = F.relu(self.deconv3(x))
        # reconstruction = torch.sigmoid(self.deconv4(x))
        reconstruction = x
        return reconstruction

class Encoder(nn.Module):
    """ VAE encoder """
    def __init__(self, img_channels, latent_size, args):
        super(Encoder, self).__init__()
        self.latent_size = latent_size
        #self.img_size = img_size
        self.img_channels = img_channels

        self.conv1 = nn.Conv2d(img_channels, 128, [1, 200], stride=1)
        # self.conv2 = nn.Conv2d(32, 64, [1, 16], stride=[1, 2])
        # self.conv3 = nn.Conv2d(64, 128, [1, 16], stride=[1, 2])
        # self.conv4 = nn.Conv2d(128, 128, [1, 8], stride=[1, 2])

        if args.type == 'dtse_test0_1':
            global enc_out
            enc_out = 512
        self.fc_mu = nn.Linear(enc_out, latent_size)
        self.fc_logsigma = nn.Linear(enc_out, latent_size)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        # x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)

        mu = self.fc_mu(x)
        logsigma = self.fc_logsigma(x)

        return mu, logsigma
