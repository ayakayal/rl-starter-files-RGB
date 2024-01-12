import math
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical


class ICMModule():
    def __init__(self, device, emb_network, inv_network, forw_network, learning_rate = 0.0001):

        self.device = device

        # pathak used Beta = 0.2 and landa=0.1
        self.forward_loss_coef = 10
        self.inverse_loss_coef = 0.1
        # networks
        self.state_embedding = emb_network
        self.inverse_dynamics = inv_network
        self.forward_dynamics = forw_network

        self.optimizer_state_embedding = optim.Adam(list(self.state_embedding.parameters()),
                                         lr=learning_rate)
        self.optimizer_inverse_dynamics = optim.Adam(list(self.inverse_dynamics.parameters()),
                                         lr=learning_rate)
        self.optimizer_forward_dynamics = optim.Adam(list(self.forward_dynamics.parameters()),
                                         lr=learning_rate)
        # move to GPU/CPU
        self.state_embedding = self.state_embedding.to(self.device)
        self.inverse_dynamics = self.inverse_dynamics.to(self.device)
        self.forward_dynamics = self.forward_dynamics.to(self.device)

        self.state_embedding.train()
        self.inverse_dynamics.train()
        self.forward_dynamics.train()

    def preprocess_observations(self,input_obs):

        # check if it is numpy or tensor and convert to tensor
        if not torch.is_tensor(input_obs):
            input_obs = torch.tensor(input_obs,device=self.device,dtype=torch.float)

        # ensure 4 dims [batch, 7, 7, 3]
        if input_obs.dim() == 3: # only one observation, is not a batch
            input_obs = input_obs.unsqueeze(0)

        # reshape to be [batch, 3, 7, 7]
        input_obs = input_obs.transpose(1, 3).transpose(2, 3)

        return input_obs

    def compute_intrinsic_reward(self,obs,next_obs,actions):
        """
            Genrate Intrinsic reward bonus based on the given input
        """
        # print('Compute Intrinsic Reward At ICM Module')
        # get tensor shape [batch,3,7,7]
        input_obs = self.preprocess_observations(obs)
        input_next_obs = self.preprocess_observations(next_obs)

        with torch.no_grad():
            # s' after embedding network
            next_state_emb = self.state_embedding(input_next_obs)

            # state embedding of s for forward dynamic model
            state_emb = self.state_embedding(input_obs) # returns [batch,feat_output_dim=32]
            # prediction of s' taken into account also actual environment action
            act = actions.unsqueeze(0).unsqueeze(0) # add dimensions for the scalar to become a "list"; then add batch dim
            pred_next_state_emb = self.forward_dynamics(state_emb,act)
            # print('state emb {}; pred_next_state: {}'.format(state_emb.shape,pred_next_state_emb.shape))


        # Calculate intrinsic rewards; we get [batch, num_feature_prediction]
        intrinsic_reward = torch.norm(pred_next_state_emb - next_state_emb, dim=1, p=2)
        # print('INt rw new:',intrinsic_reward)

        return intrinsic_reward.item()

    def update(self,obs,next_obs,actions):
        """
            Update NN parameters with batch of observations
        """
        # get tensor shape [batch,3,7,7]
        input_obs = self.preprocess_observations(obs)
        input_next_obs = self.preprocess_observations(next_obs)

        # get s and s'
        state_emb = self.state_embedding(input_obs)
        next_state_emb = self.state_embedding(input_next_obs)

        # *********************************************************************
        # 1. loss of inverse module
        pred_actions = self.inverse_dynamics(state_emb, next_state_emb)
        true_actions = actions

        # pre-process
        log_soft_pred_actions = F.log_softmax(pred_actions,dim=1)
        true_actions = true_actions.long() # int tensor type; Long is required for nll_loss
        #print('log_soft_pref_actions:',log_soft_pred_actions)
        # generate cross_entropy/nll_loss
        # expected input to be:
        #  - log_soft_pred_actions --> [batch,action_size] action_size are the log probs for each action
        #  - true_actions --> [batch]
        inverse_dynamics_loss = F.nll_loss(log_soft_pred_actions,
                                           target = true_actions.flatten(),
                                           reduction='none')
        # print('cross entropy loss:', inverse_dynamics_loss.shape)

        # finally get the avg loss of the whole batch
        inverse_dynamics_loss = torch.mean(inverse_dynamics_loss, dim=0)
        # *********************************************************************

        # *********************************************************************
        # 2. loss of forward module
        pred_next_state_emb = self.forward_dynamics(state_emb,actions)
        forward_dynamics_loss = torch.norm(pred_next_state_emb - next_state_emb, dim=1, p=2)
        # print('pred next_state_embedding:',pred_next_state_emb.shape)
        # print('next_state_embedding:',next_state_emb.shape)
        # print('fwloss.shape:',forward_dynamics_loss.shape)
        forward_dynamics_loss = torch.mean(forward_dynamics_loss, dim=0)
        # *********************************************************************

        # total loss
        icm_loss = self.forward_loss_coef * forward_dynamics_loss + self.inverse_loss_coef*inverse_dynamics_loss

        # Optimization step
        self.optimizer_state_embedding.zero_grad()
        self.optimizer_inverse_dynamics.zero_grad()
        self.optimizer_forward_dynamics.zero_grad()

        # backpropagation of gradients
        icm_loss.backward()

        # grad_clipping
        torch.nn.utils.clip_grad_norm_(self.state_embedding.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.inverse_dynamics.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.forward_dynamics.parameters(), 0.5)

        self.optimizer_state_embedding.step()
        self.optimizer_inverse_dynamics.step()
        self.optimizer_forward_dynamics.step()

        return forward_dynamics_loss,inverse_dynamics_loss




# RIDE initilization
def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

# class EmbeddingNetwork_RIDE(nn.Module):
#     """
#      Based on the architectures selected at minigrid in RIDE:
#      https://github.com/facebookresearch/impact-driven-exploration/blob/877c4ea530cc0ca3902211dba4e922bf8c3ce276/src/models.py#L352    """
#     def __init__(self):
#         super().__init__()

#         input_size=7*7*3
#         init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
#                             constant_(x, 0), nn.init.calculate_gain('relu'))

#         self.feature_extractor = nn.Sequential(
#             init_(nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=2,padding=1)),
#             nn.ELU(),
#             init_(nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,stride=2,padding=1)),
#             nn.ELU(),
#             init_(nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,stride=2,padding=1)),
#             nn.ELU(),
#         )
#         # params = sum(p.numel() for p in self.modules.parameters())
#         # print('Params:',params)



#     def forward(self, next_obs):
#         feature = self.feature_extractor(next_obs)
#         reshape = feature.view(feature.size(0),-1)

#         return reshape
def init_params(m):
    
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
       

        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)
class EmbeddingNetwork_RIDE(nn.Module):
    """
     Based on the architectures selected at minigrid in RIDE:
     https://github.com/facebookresearch/impact-driven-exploration/blob/877c4ea530cc0ca3902211dba4e922bf8c3ce276/src/models.py#L352    """
    def __init__(self):
        super().__init__()

        input_size=7*7*3
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                            constant_(x, 0), nn.init.calculate_gain('relu'))

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
         
        self.apply(init_params)
        # params = sum(p.numel() for p in self.modules.parameters())
        # print('Params:',params)



    def forward(self, next_obs):
        feature = self.feature_extractor(next_obs)
        #reshape = feature.view(feature.size(0),-1)
        embedding = feature.reshape(feature.shape[0], -1)

        return embedding

class InverseDynamicsNetwork_RIDE(nn.Module):
    def __init__(self, num_actions, input_size = 64, device = 'cuda'): #input size was 32 and device was cpu
        super().__init__()
        self.input_size = input_size
        self.num_actions = num_actions
        self.device = device

        init_ = lambda m: init(m, nn.init.orthogonal_,
                lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('relu'))
        self.inverse_dynamics = nn.Sequential(
            init_(nn.Linear(2 * self.input_size, 256)),
            nn.ReLU(),
        )

        init_ = lambda m: init(m, nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0))
        self.id_out = init_(nn.Linear(256, self.num_actions))


    def forward(self, state_embedding, next_state_embedding):
        inputs = torch.cat((state_embedding, next_state_embedding), dim=1)

        action_logits = self.id_out(self.inverse_dynamics(inputs))
        return action_logits


class ForwardDynamicsNetwork_RIDE(nn.Module):
    def __init__(self, num_actions, input_size=64, device = 'cuda'): #input size was 32 and the device was cpu
        """
            input_size depends on the output of the embedding network
        """
        super().__init__()
        self.input_size = input_size
        self.num_actions = num_actions
        self.device = device


        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                            constant_(x, 0), nn.init.calculate_gain('relu'))

        self.forward_dynamics = nn.Sequential(
            init_(nn.Linear(self.input_size + self.num_actions, 256)),
            nn.ReLU(),
        )

        init_ = lambda m: init(m, nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0))

        # the loss of ICM is between two latent predictions; therefore, they have to have the same dimensiones
        # as the Embedding Network output is used here as input, we use it to determine this NN output
        self.fd_out = init_(nn.Linear(256, self.input_size))

    def forward(self, state_embedding, action):
        """
            INPUTS:
            -Action: it can be a single item value (when computing rewards) or
                    batch of values when updating [batch,action_taken]
            -State-embedding:
        """
        actions_one_hot = torch.zeros((action.shape[0],self.num_actions), device=self.device)

        # generate one-hot encoding of action
        for i,a in enumerate(action):
            a = a.squeeze(0) # we need a scalar tensor value, not a list for the transformation of one-hot
            a = a.long()# the transformation requires to be Long type
            one_hot = F.one_hot(a, num_classes=self.num_actions).float()
            actions_one_hot[i].copy_(one_hot)

        # concat and generate the input
        inputs = torch.cat((state_embedding, actions_one_hot),dim=1)
        # forward pass
        next_state_emb = self.fd_out(self.forward_dynamics(inputs))

        return next_state_emb

