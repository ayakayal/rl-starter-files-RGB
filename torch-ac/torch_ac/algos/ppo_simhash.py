import torch
import torch.nn.functional as F

from torch_ac.format import default_preprocess_obss
from torch_ac.utils import DictList, ParallelEnv
import numpy as np
from torch_ac.algos import PPOAlgo

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import minigrid.core
    
# SimHash with better state representation for the case of Doorkey only, appending whether the key is picked up or not to the hashed key
class EmbeddingNetwork(nn.Module):
    """
     Based on the architectures selected at minigrid in RIDE:
     https://github.com/facebookresearch/impact-driven-exploration/blob/877c4ea530cc0ca3902211dba4e922bf8c3ce276/src/models.py#L352    """
    def __init__(self,input_dim=40000,output_dim=512):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )  
        self.fc_layer = nn.Linear(input_dim, output_dim)  
    def forward(self, obs):
        obs=obs.image.transpose(1, 3).transpose(2, 3)
        feature = self.feature_extractor(obs)
        reshape = feature.reshape(feature.shape[0], -1)
        output = self.fc_layer(reshape)

        return output


class SimHash(object) :
  def __init__(self, state_emb_size, k , state_emb_function,device) :
    ''' Hashing between continuous state space and discrete state space '''
    self.hash = {}
    #self.A = np.random.normal(0,1, (k , state_emb_size)) #on the CPU
    self.A = torch.normal(0,1, (k , state_emb_size)).to('cuda') #comment for method 2
    #self.A = torch.normal(0,1, (k , state_emb_size+1)).to('cuda') # method1 the +1 dimension here is to account for the key
    self.device = device
    self.embedding_network=state_emb_function

  def count(self, preprocessed_states,key_picked_up_tensor) :
    ''' Increase the count for the states and retourn the counts '''
    counts = []
    #state embedding phase
    #print('key_picked_up_tensor',key_picked_up_tensor)
    state_embeddings=self.embedding_network(preprocessed_states)
    #print('state_embeddings',state_embeddings.shape)
    #state_embeddings_with_key_info = torch.cat((state_embeddings, key_picked_up_tensor.unsqueeze(1)), dim=1) #method 1
    #print(' state_embeddings_with_key_info', state_embeddings_with_key_info[:,512])
    keys_before_picked_up_info = torch.matmul(self.A, state_embeddings.t()) #uncomment for method 2
    keys = torch.cat((keys_before_picked_up_info.t(), key_picked_up_tensor.unsqueeze(1)), dim=1) #uncomment for method 3
    keys = torch.sign(keys) #for method 2
    #keys = torch.sign(torch.matmul(self.A, state_embeddings_with_key_info.t())) #uncomment for method 1
    # print('keys shape',keys.shape)

   
    for key in keys: #it was keys.t() for method 1
        #key = tuple(key.tolist()) 
        # print('key',key) # Convert to tuple for dictionary key
        key=tuple(key.tolist()) #necessary to make it uniquely hashable
        if key in self.hash:
            self.hash[key] += 1
        else:
            self.hash[key] = 1
        counts.append(self.hash[key])

    return torch.tensor(counts, device=self.device)



class PPOAlgoStateHash(PPOAlgo):
    def __init__(self, envs, acmodel, device=None, num_frames_per_proc=None, discount=0.99, lr=0.001, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 adam_eps=1e-8, clip_eps=0.2, epochs=4, batch_size=256,singleton_env=False, RGB=False, preprocess_obss=None,intrinsic_reward_coeff=0.0001,k=256,
                 reshape_reward=None): 

        super().__init__(envs, acmodel, device, num_frames_per_proc, discount, lr, gae_lambda,
                 entropy_coef, value_loss_coef, max_grad_norm, recurrence,
                 adam_eps, clip_eps, epochs, batch_size, singleton_env,RGB, preprocess_obss,
                 reshape_reward)
        
        self.intrinsic_reward_coeff = intrinsic_reward_coeff
        
       
           
        shape = (self.num_frames_per_proc, self.num_procs)
        self.total_rewards= torch.zeros(*shape, device=self.device)

        #initialize intrinsic rewards
        self.intrinsic_rewards=torch.zeros(*shape, device=self.device)
        self.log_episode_return_int = torch.zeros(self.num_procs, device=self.device)
        self.log_return_int =  [0] * self.num_procs
        self.hash_function= SimHash(state_emb_size=512,k=k,state_emb_function= EmbeddingNetwork().to(self.device),device=self.device)


        
    def pass_models_parameters(self):
        return self.train_state_count
    
    def collect_experiences(self):
        shape = (self.num_frames_per_proc, self.num_procs)
        self.total_rewards= torch.zeros(*shape, device=self.device)

        #initialize intrinsic rewards
        self.intrinsic_rewards=torch.zeros(*shape, device=self.device)
  
        for i in range(self.num_frames_per_proc):
            self.total_frames+=self.num_procs
            
            # Do one agent-environment interaction
            preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
            with torch.no_grad():
                if self.acmodel.recurrent:
                    dist, value, memory = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
                else:
                    dist, value = self.acmodel(preprocessed_obs)
            action = dist.sample()
            if self.vizualise_video==True:
                self.frames.append(np.moveaxis(self.env.envs[0].get_frame(), 2, 0))
           
            obs, reward, terminated, truncated, agent_loc, _ = self.env.step(action.cpu().numpy())
            for agent_state in agent_loc:
                if agent_state in self.state_visitation_pos.keys():
                    self.state_visitation_pos[agent_state] += 1
                else:
                    self.state_visitation_pos[agent_state] = 1

            for r in reward:
                if r!=0 and self.found_reward==0:
                    self.saved_frame_first_reward=self.total_frames
                    self.found_reward=1
                    continue
                if r!=0 and self.found_reward==1:
                    self.saved_frame_second_reward=self.total_frames
                    self.found_reward=2
                    continue
                if r!=0 and self.found_reward==2:
                    self.saved_frame_third_reward=self.total_frames
                    self.found_reward=3
                    continue
           
            temp_irewards=torch.zeros(self.num_procs, device=self.device).detach()
            
            
            preprocessed_states = self.preprocess_obss(obs, device=self.device)
            key_picked_up_list=[]
            for env in self.env.envs:
                key_picked_up= True #assume key is picked up
                # print(env.grid.grid)
                for item in env.grid.grid:
    
                    if item and isinstance(item, minigrid.core.world_object.Key): #if key is there then it's not pickedup
                    
                        key_picked_up = False
                        break
                # if key_picked_up == True:
                #     print('key is picked_up')
                key_picked_up_list.append(key_picked_up)
            key_picked_up_tensor = torch.tensor(key_picked_up_list, device=self.device)
            with torch.no_grad():
                counts=self.hash_function.count(preprocessed_states,key_picked_up_tensor )
               

            temp_irewards= self.intrinsic_reward_coeff / torch.sqrt(counts)
            self.intrinsic_rewards[i]=temp_irewards.clone().detach()
            self.intrinsic_reward_per_frame=(torch.mean(self.intrinsic_rewards[i])).item()
            done = tuple(a | b for a, b in zip(terminated, truncated))

            for p in range(self.num_procs):
                obs_tuple=tuple( obs[p]['image'].reshape(-1).tolist())
                if obs_tuple in self.train_state_count:
                    self.train_state_count[obs_tuple]+= 1
                else:
                    self.train_state_count[obs_tuple]=1


            self.obss[i] = self.obs
            self.obs = obs
            if self.acmodel.recurrent:
                self.memories[i] = self.memory
                self.memory = memory
            self.masks[i] = self.mask  
            self.mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
            self.actions[i] = action
            self.values[i] = value
            if self.reshape_reward is not None:
                self.rewards[i] = torch.tensor([
                    self.reshape_reward(obs_, action_, reward_, done_)
                    for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
                ], device=self.device)
            else:
                self.rewards[i] = torch.tensor(reward, device=self.device)
            self.total_rewards[i]= self.intrinsic_rewards[i] + self.rewards[i]
            self.total_rewards[i] /= (1+self.intrinsic_reward_coeff)

            if self.singleton_env != 'False':
     
                for idx in range(len(self.intrinsic_rewards[i])):
        
                    if agent_loc[idx] in self.ir_dict.keys():
         
                        self.ir_dict[agent_loc[idx]] += self.intrinsic_rewards[i][idx].item()
                    else:
                        self.ir_dict[agent_loc[idx]] = self.intrinsic_rewards[i][idx].item()

            self.log_probs[i] = dist.log_prob(action)
            # Update log values

            self.log_episode_return += torch.tensor(reward, device=self.device, dtype=torch.float)
            self.log_episode_return_int += self.intrinsic_rewards[i].clone().detach()
            self.log_episode_reshaped_return += self.rewards[i]
            self.log_episode_num_frames += torch.ones(self.num_procs, device=self.device)

            for i, done_ in enumerate(done):
                if done_:

                    self.log_done_counter += 1
                    self.log_return.append(self.log_episode_return[i].item())
                    self.log_reshaped_return.append(self.log_episode_reshaped_return[i].item())
                    self.log_num_frames.append(self.log_episode_num_frames[i].item())
                    self.log_return_int.append(self.log_episode_return_int[i].item())

                    

            self.log_episode_return *= self.mask #to reset when the episode is done
            self.log_episode_reshaped_return *= self.mask
            self.log_episode_num_frames *= self.mask
            self.log_episode_return_int *= self.mask

        # Add advantage and return to experiences

        preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
        with torch.no_grad():
            if self.acmodel.recurrent:
                _, next_value, _ = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
            else:
                _, next_value = self.acmodel(preprocessed_obs)

        for i in reversed(range(self.num_frames_per_proc)):
            next_mask = self.masks[i+1] if i < self.num_frames_per_proc - 1 else self.mask
            next_value = self.values[i+1] if i < self.num_frames_per_proc - 1 else next_value
            next_advantage = self.advantages[i+1] if i < self.num_frames_per_proc - 1 else 0
            delta = self.total_rewards[i] + self.discount * next_value * next_mask - self.values[i]
 
            self.advantages[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask #it was delta instead of delta_intrinsic


        # Define experiences:
        #   the whole experience is the concatenation of the experience
        #   of each process.
        # In comments below:
        #   - T is self.num_frames_per_proc,
        #   - P is self.num_procs,
        #   - D is the dimensionality.

        exps = DictList()
        exps.obs = [self.obss[i][j]
                    for j in range(self.num_procs)
                    for i in range(self.num_frames_per_proc)]
        if self.acmodel.recurrent:
            # T x P x D -> P x T x D -> (P * T) x D
            exps.memory = self.memories.transpose(0, 1).reshape(-1, *self.memories.shape[2:])
            # T x P -> P x T -> (P * T) x 1
            exps.mask = self.masks.transpose(0, 1).reshape(-1).unsqueeze(1)
        # for all tensors below, T x P -> P x T -> P * T
        exps.action = self.actions.transpose(0, 1).reshape(-1)
        exps.value = self.values.transpose(0, 1).reshape(-1)
        exps.reward = self.rewards.transpose(0, 1).reshape(-1)
        exps.reward_total=self.total_rewards.transpose(0,1).reshape(-1)
        exps.advantage = self.advantages.transpose(0, 1).reshape(-1)
        exps.returnn = exps.value + exps.advantage #this include the intrinsic reward in the advantage
        exps.log_prob = self.log_probs.transpose(0, 1).reshape(-1)

        # Preprocess experiences

        exps.obs = self.preprocess_obss(exps.obs, device=self.device)

        # Log some values

        keep = max(self.log_done_counter, self.num_procs)
        self.number_of_visited_states= len(self.train_state_count)
         #size of the grid and the possible combinations of object index, color and status
        self.state_coverage= self.number_of_visited_states #percentage of state coverage
        non_zero_count=0
        for key, value in self.state_visitation_pos.items():
            if value != 0:
                non_zero_count += 1
        self.state_coverage_position= non_zero_count
       

        logs = {
            "return_per_episode": self.log_return[-keep:],
            "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "num_frames": self.num_frames,
            "return_int_per_episode": self.log_return_int[-keep:],
            "state_coverage": self.state_coverage, #partial observations count
            "frame_first_reward": self.saved_frame_first_reward,
            "frame_second_reward": self.saved_frame_second_reward,
            "frame_third_reward": self.saved_frame_third_reward,
            "state_visitation_pos":self.state_visitation_pos, #dictionary of state positions count
            "state_coverage_position":self.state_coverage_position, # count of how many (x,y) positions are covered
            "reward_int_per_frame":self.intrinsic_reward_per_frame,
            "ir_dict":self.ir_dict,
            "hash_dict":len(self.hash_function.hash)

            
        }

        self.log_done_counter = 0
        self.log_return = self.log_return[-self.num_procs:]
        self.log_return_int = self.log_return_int[-self.num_procs:]
        self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
        self.log_num_frames = self.log_num_frames[-self.num_procs:]

        return exps, logs, self.frames 
    