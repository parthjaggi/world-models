"""
Simulated carracing environment.
"""
from os.path import join, exists
import torch
from torch.distributions.categorical import Categorical
import gym
from gym import spaces
from models.vae import VAE
from models.mdrnn import MDRNNCell

import numpy as np

class SimulatedCarracing(gym.Env): # pylint: disable=too-many-instance-attributes
    """
    Simulated Car Racing
    """
    LSIZE = 32
    HSIZE = 256
    STATE_H = 64
    STATE_W = 64

    def __init__(self, directory):
        vae_file = join(directory, 'vae', 'best.tar')
        rnn_file = join(directory, 'mdrnn', 'best.tar')
        assert exists(vae_file), "No VAE model in the directory..."
        assert exists(rnn_file), "No MDRNN model in the directory..."

        # spaces
        self.action_space = spaces.Box(np.array([-1, 0, 0]), np.array([1, 1, 1]))
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.STATE_H, self.STATE_W, 3),
                                            dtype=np.uint8)

        # load VAE
        vae = VAE(3, self.LSIZE)
        vae_state = torch.load(vae_file, map_location=lambda storage, location: storage)
        print("Loading VAE at epoch {}, "
              "with test error {}...".format(
                  vae_state['epoch'], vae_state['precision']))
        vae.load_state_dict(vae_state['state_dict'])
        self._decoder = vae.decoder

        # load MDRNN
        self._rnn = MDRNNCell(32, 3, self.HSIZE, 5)
        rnn_state = torch.load(rnn_file, map_location=lambda storage, location: storage)
        print("Loading MDRNN at epoch {}, "
              "with test error {}...".format(
                  rnn_state['epoch'], rnn_state['precision']))
        rnn_state_dict = {k.strip('_l0'): v for k, v in rnn_state['state_dict'].items()}
        self._rnn.load_state_dict(rnn_state_dict)

        # init state
        self._lstate = torch.randn(1, self.LSIZE)
        self._hstate = 2 * [torch.zeros(1, self.HSIZE)]

        # obs
        self._obs = None
        self._visual_obs = None

        # rendering
        self.monitor = None
        self.figure = None

    def reset(self):
        """ Resetting """
        import matplotlib.pyplot as plt
        self._lstate = torch.randn(1, self.LSIZE)
        self._hstate = 2 * [torch.zeros(1, self.HSIZE)]

        # also reset monitor
        if not self.monitor:
            self.figure = plt.figure()
            self.monitor = plt.imshow(
                np.zeros((self.STATE_H, self.STATE_W, 3),
                         dtype=np.uint8))

    def step(self, action):
        """ One step forward """
        with torch.no_grad():
            action = torch.Tensor(action).unsqueeze(0)
            mu, sigma, pi, r, d, n_h = self._rnn(action, self._lstate, self._hstate)
            pi = pi.squeeze()
            mixt = Categorical(pi).sample().item()

            self._lstate = mu[:, mixt, :] + sigma[:, mixt, :] * torch.randn_like(mu[:, mixt, :])
            self._hstate = n_h

            self._obs = self._decoder(self._lstate)
            np_obs = self._obs.numpy()
            np_obs = np.clip(np_obs, 0, 1) * 255
            np_obs = np.transpose(np_obs, (0, 2, 3, 1))
            np_obs = np_obs.squeeze()
            np_obs = np_obs.astype(np.uint8)
            self._visual_obs = np_obs

            return np_obs, r.item(), d.item() > 0

    def render(self): # pylint: disable=arguments-differ
        """ Rendering """
        import matplotlib.pyplot as plt
        if not self.monitor:
            self.figure = plt.figure()
            self.monitor = plt.imshow(
                np.zeros((self.STATE_H, self.STATE_W, 3),
                         dtype=np.uint8))
        self.monitor.set_data(self._visual_obs)
        plt.pause(.01)

if __name__ == '__main__':
    env = SimulatedCarracing('logs/exp0')
    env.reset()
    action = np.array([0., 0., 0.])

    def on_key_press(event):
        """ Defines key pressed behavior """
        if event.key == 'up':
            action[1] = 1
        if event.key == 'down':
            action[2] = .8
        if event.key == 'left':
            action[0] = -1
        if event.key == 'right':
            action[0] = 1

    def on_key_release(event):
        """ Defines key pressed behavior """
        if event.key == 'up':
            action[1] = 0
        if event.key == 'down':
            action[2] = 0
        if event.key == 'left' and action[0] == -1:
            action[0] = 0
        if event.key == 'right' and action[0] == 1:
            action[0] = 0

    env.figure.canvas.mpl_connect('key_press_event', on_key_press)
    env.figure.canvas.mpl_connect('key_release_event', on_key_release)
    while True:
        _, _, done = env.step(action)
        env.render()
        if done:
            break