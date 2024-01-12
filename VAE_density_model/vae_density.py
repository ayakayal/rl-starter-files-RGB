import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import utils.pytorch_util as ptu
# from core import PyTorchModule
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        #print('oui')
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)
        #print("weight data type", m.weight.data.dtype)


class VAEDensity(nn.Module):
    def __init__(self,
                 input_size,
                 num_skills=0,
                 code_dim=128,
                 beta=0.5,
                 lr=1e-3,
                 ):
        """Initialize the density model.

        Args:
          num_skills: number of densities to simultaneously track
        """
        # self.save_init_params(locals())
        super().__init__()
        torch.autograd.set_detect_anomaly(True)
        self._num_skills = num_skills

        input_dim = np.prod(input_size)
        #print('input_dim',input_dim)
        self.enc = nn.Sequential(
            nn.Linear(input_dim, 150),
            nn.ReLU(),
            nn.Linear(150, 150),
            nn.ReLU(),
        )
        self.enc_mu = nn.Linear(150, code_dim)
        self.enc_logvar = nn.Linear(150, code_dim)
        self.dec = nn.Sequential(
            nn.Linear(code_dim, 150),
            nn.ReLU(),
            nn.Linear(150, 150),
            nn.ReLU(),
            nn.Linear(150, input_dim),
        )
        self.apply(init_params)
        self.lr = lr
        self.beta = beta
        params = (list(self.enc.parameters()) +
                  list(self.enc_mu.parameters()) +
                  list(self.enc_logvar.parameters()) +
                  list(self.dec.parameters()))
        self.optimizer = optim.Adam(params, lr=self.lr)

    def get_output_for(self, aug_obs, sample=True):
        """
        Returns the log probability of the given observation.
        """
        obs = aug_obs
        print(obs)
        with torch.no_grad():
            enc_features = self.enc(obs)
            print('enc_features',enc_features)
            mu = self.enc_mu(enc_features)
            logvar = self.enc_logvar(enc_features)

            stds = (0.5 * logvar).exp()
            if sample:
                # epsilon = ptu.randn(*mu.size())
                epsilon = torch.randn(*mu.size(),device='cuda')
            else:
                epsilon = torch.ones_like(mu,device='cuda')
            print('epsilon',epsilon)
            print('stds',stds)
            print('mu',mu)
            code = epsilon * stds + mu

            obs_distribution_params = self.dec(code)
            log_prob = -1. * F.mse_loss(obs, obs_distribution_params,
                                        reduction='none') #it was none
            log_prob = torch.sum(log_prob, -1, keepdim=True)
        return log_prob.detach()

    def update(self, aug_obs):
        obs = aug_obs

        enc_features = self.enc(obs)
        mu = self.enc_mu(enc_features)
        logvar = self.enc_logvar(enc_features)

        stds = (0.5 * logvar).exp()
        # epsilon = ptu.randn(*mu.size())
        epsilon = torch.randn(*mu.size(),device='cuda')
        code = epsilon * stds + mu

        kle = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp(), dim=1
        ).mean()

        obs_distribution_params = self.dec(code)
        log_prob = -1. * F.mse_loss(obs, obs_distribution_params,
                                    reduction='mean') #it was mean

        loss = self.beta * kle - log_prob

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.cpu().item()