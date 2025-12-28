import numpy as np

import torch
import torch.nn as nn
import random
from torch.distributions.categorical import Categorical

from .common import Predictor, layer_init


class PPODefault(Predictor):
    def __init__(self, envs, device, observation_shape=None):
        super().__init__()
        self.device = device
        if observation_shape is None:
            dims = envs.observation_space.shape
        else:
            dims = observation_shape

        network_conv = nn.Sequential(
            layer_init(nn.Conv2d(dims[0], 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            print(dims)
            dummy_input = torch.zeros(dims)
            print(torch.flatten(network_conv(dummy_input)).shape)
            conv_out_size = torch.flatten(network_conv(dummy_input)).shape[0]

        self.network = nn.Sequential(
            layer_init(nn.Conv2d(dims[0], 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(conv_out_size, 512)),
            nn.ReLU(),
        )

        self.actor = layer_init(nn.Linear(512, envs.action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

    def forward(self, x, head=None):
        if head == "actor":
            hidden = self.network(x / 255.0)
            logits = self.actor(hidden)
            probs = Categorical(logits=logits)
            action = probs.sample()
            return probs.log_prob(action)
        elif head == "critic":
            return self.critic(self.network(x / 255.0))
        else:
            raise ValueError(f"Unknown head: {head}")


class PPODefault_old(Predictor):
    def __init__(self, envs, device):
        super().__init__()
        self.device = device

        dims = envs.observation_space.shape

        self.network = nn.Sequential(
            layer_init(nn.Conv2d(dims[0], 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, envs.action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

    def forward(self, x, head=None):
        if head == "actor":
            hidden = self.network(x / 255.0)
            logits = self.actor(hidden)
            probs = Categorical(logits=logits)
            action = probs.sample()
            return probs.log_prob(action)
        elif head == "critic":
            return self.critic(self.network(x / 255.0))
        else:
            raise ValueError(f"Unknown head: {head}")


class PPOScaled(Predictor):
    def __init__(self, envs, device):
        super().__init__()
        self.device = device
        dims = envs.observation_space.shape

        self.network_conv = nn.Sequential(
            layer_init(nn.Conv2d(dims[0], 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            print(envs.observation_space.shape)
            dummy_input = torch.zeros(envs.observation_space.shape)
            print(torch.flatten(self.network_conv(dummy_input)).shape)
            conv_out_size = torch.flatten(self.network_conv(dummy_input)).shape[0]
        self.network = nn.Sequential(
            self.network_conv,
            layer_init(nn.Linear(conv_out_size, 512)),
            nn.ReLU(),
        )

        self.actor = layer_init(nn.Linear(512, envs.action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


class PPObj(Predictor):
    def __init__(self, envs, device, encoder_dims=(128, 64), decoder_dims=(32,)):
        super().__init__()
        self.device = device

        dims = envs.observation_space.shape
        layers = nn.ModuleList()

        in_dim = dims[-1]

        for l in encoder_dims:
            layers.append(layer_init(nn.Linear(in_dim, l)))
            layers.append(nn.ReLU())
            in_dim = l
        layers.append(nn.Flatten())
        in_dim *= np.prod(dims[:-1], dtype=int)
        l = in_dim
        for l in decoder_dims:
            layers.append(layer_init(nn.Linear(in_dim, l)))
            layers.append(nn.ReLU())
            in_dim = l

        self.network = nn.Sequential(*layers)
        self.actor = layer_init(nn.Linear(l, envs.action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(l, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


class PPO_Obj(Predictor):
    def __init__(self, envs, device):
        super().__init__()
        self.device = device

        self.network = nn.Sequential(
            layer_init(nn.Linear(4, 32)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, envs.action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


class PPOCombi2Big(Predictor):
    def __init__(self, envs, device, encoder_dims=(256, 512, 1024, 512), decoder_dims=(512,), normalize_obj=True,
                 use_masks=True, use_obj=True):
        super().__init__()
        self.device = device
        self.normalize_obj = normalize_obj
        self.use_masks = use_masks
        self.use_obj = use_obj
        print(envs.observation_space)
        print("encoder_dims", encoder_dims)
        print("decoder_dims", decoder_dims)
        self.decoder_dims = decoder_dims
        dims_obj = envs.observation_space["obj"].shape
        dims_masks = envs.observation_space["masks"].shape
        self.network_conv = nn.Sequential(
            layer_init(nn.Conv2d(dims_masks[0], 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            print(envs.observation_space["masks"].shape)
            dummy_input = torch.zeros((1,) + envs.observation_space["masks"].shape)
            print(torch.flatten(self.network_conv(dummy_input)).shape)
            self.conv_out_size = torch.flatten(self.network_conv(dummy_input)).shape[0]

        layers = nn.ModuleList()

        in_dim = dims_obj[-1]

        for l in encoder_dims:
            layers.append(layer_init(nn.Linear(in_dim, l)))
            layers.append(nn.ReLU())
            in_dim = l
        layers.append(nn.Flatten())
        in_dim *= np.prod(dims_obj[:-1], dtype=int)
        l = in_dim
        for l in decoder_dims:
            layers.append(layer_init(nn.Linear(in_dim, l)))
            layers.append(nn.ReLU())
            in_dim = l

        self.network_linear = nn.Sequential(*layers)
        ''' 
        self.network_cat = nn.Sequential(
            layer_init(nn.Linear(self.conv_out_size + self.decoder_dims[-1], 512)),  # + 32, 512 * 2)),
            nn.LayerNorm(512),
            nn.ReLU(),
        )
        '''
        self.network_cat = nn.Sequential(
            layer_init(nn.Linear(self.conv_out_size + self.decoder_dims[-1], 1024)),
            nn.ReLU(),
            layer_init(nn.Linear(1024, 512)),
            nn.ReLU(),
        )

        self.actor = layer_init(nn.Linear(512, envs.action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_hidden(self, x):
        if self.use_masks:
            x_masks = x["masks"] / 255.0
        else:
            x_masks = torch.zeros_like(x["masks"])
        if self.use_obj:
            if self.normalize_obj:
                x_obj = x["obj"] / 210.0
            else:
                x_obj = x["obj"]
        else:
            x_obj = torch.zeros_like(x["obj"])
        hidden_conv = self.network_conv(x_masks)
        hidden_linear = self.network_linear(x_obj)
        hidden_flat = torch.cat((hidden_conv, hidden_linear), dim=1)
        hidden = self.network_cat(hidden_flat)
        return hidden
        # return hidden_linear

    def get_value(self, x):
        return self.critic(self.get_hidden(x))

    def get_action_and_value(self, x, action=None):
        hidden = self.get_hidden(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

