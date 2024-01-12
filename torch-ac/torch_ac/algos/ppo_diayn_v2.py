import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_ac
from torch_ac.format import default_preprocess_obss
from torch_ac.utils import DictList, ParallelEnv
import numpy as np
from torch_ac.algos import PPOAlgo
from torch_ac.utils.clip_grads import global_grad_norm_
from diayn_models import DIAYN_discriminator

# def init_params(m):
#     classname = m.__class__.__name__
#     if classname.find("Linear") != -1:
#         #print('oui')
#         m.weight.data.normal_(0, 1)
#         m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
#         if m.bias is not None:
#             m.bias.data.fill_(0)

class DIAYN_reward():
    def __init__(self,no_skills,discriminator,beta):
        self.discriminator = discriminator
        self.no_skills = no_skills
        # Each skill equally likely to be chosen
        self.prior_probability_of_skill = torch.tensor(1/no_skills)

        self.beta = beta

    def get_predicted_probability_of_skill(self, skill, next_state):
        """
        Gets the probability that the discriminator gives to the correct skill and also returns the full
        unnormalised probabilities vector which is the output of the discriminator network.
        """
        predicted_probabilities_unnormalised = self.discriminator(next_state)
        probability_of_correct_skill = F.softmax(predicted_probabilities_unnormalised)[:, skill]

        return  probability_of_correct_skill.item(), predicted_probabilities_unnormalised

    def diayn_reward(self, probability_correct_skill):
        """
        Calculates an intrinsic reward that encourages maximum exploration. It also keeps track of the discriminator
        outputs so they can be used for training
        """

        # calculate rewards as log q - log p
        intrinsic_reward = (torch.log(probability_correct_skill) - torch.log(self.prior_probability_of_skill+ 1e-6))

        return intrinsic_reward.item()
    
    def sample_skills(self,no_envs):
        """
        Sample a skill for each environment (uniform sampling)
        """
        skills = np.random.randint(0, self.no_skills-1, no_envs)

        return skills

    
    def hot_encode_skills(self,skills):
        """
        Hot encode skills to be fed to actor-critic networks
        """
        
        # Create a zero tensor of shape [batch_size, num_skills]
        skills = skills.to(torch.int64)
        # print('skills.device',skills.device)
        one_hot_skills = torch.zeros((skills.shape[0], self.no_skills), device='cuda')
        # Set the indices of the skills to 1
        one_hot_skills.scatter_(1, skills.unsqueeze(1), 1)
        # print('one_hot_skills',one_hot_skills)
        return one_hot_skills



# class DIAYN_discriminator(nn.Module,torch_ac.RecurrentACModel):
#     def __init__(self, no_skills, use_memory=False, use_text=False):
#         super().__init__()

#         # Define image embedding
#         self.image_conv = nn.Sequential(
#             nn.Conv2d(3, 16, (2, 2)),
#             nn.ReLU(),
#             nn.MaxPool2d((2, 2)),
#             nn.Conv2d(16, 32, (2, 2)),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, (2, 2)),
#             nn.ReLU()
#         )
#         n = 56
#         m = 56
#         self.image_embedding_size = 40000

#         # Resize image embedding
#         self.embedding_size = self.semi_memory_size

#         # Define the discriminator network
#         self.discriminator = nn.Sequential(
#             nn.Linear(self.embedding_size, 64),
#             nn.Tanh(),
#             nn.Linear(64, no_skills)
#         )        
        
#         # Initialize parameters correctly
#         self.apply(init_params)

#     @property
#     def memory_size(self):
#         return 2*self.semi_memory_size

#     @property
#     def semi_memory_size(self):
#         return self.image_embedding_size
    
#     def forward(self,next_state):
#         """
#         The forward takes as input the next_state and returns an unnormalized prob over skills latent space.
#         """

#         # apply the 3D CNN to the image 
#         embedding = next_state.image.transpose(1, 3).transpose(2, 3)
#         embedding = self.image_conv(embedding)
#         embedding = embedding.reshape(embedding.shape[0], -1)

#         # return an unnormalized distribution over the skills
#         x = self.discriminator(embedding)

#         return x


class PPOAlgoDIAYNV2(PPOAlgo):
    def __init__(self, envs, acmodel, device=None, num_frames_per_proc=None, discount=0.99, lr=0.001, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 adam_eps=1e-8, clip_eps=0.2, epochs=4, batch_size=256,singleton_env=False, RGB=False, preprocess_obss=None,intrinsic_reward_coeff=0.0001, num_skills=4,disc_lr=0.0001,pretraining=False,reshape_reward=None): 
      
        super().__init__(envs, acmodel, device, num_frames_per_proc, discount, lr, gae_lambda,
                 entropy_coef, value_loss_coef, max_grad_norm, recurrence,
                 adam_eps, clip_eps, epochs, batch_size,singleton_env, RGB,preprocess_obss,
                 reshape_reward)
        self.no_skills=num_skills
        print('self.no_skills',self.no_skills)
        self.intrinsic_reward_coeff = intrinsic_reward_coeff
        self.diayn_discriminator = DIAYN_discriminator(self.no_skills)
        self.diayn_discriminator.to(self.device)
            # DIAYN reward class
        self.diayn_reward = DIAYN_reward(self.no_skills, self.diayn_discriminator,self.intrinsic_reward_coeff)
        combined_params = list(self.acmodel.parameters()) + list(self.diayn_discriminator.parameters())
        self.optimizer = torch.optim.Adam(combined_params, lr, eps=adam_eps) 
        
        shape = (self.num_frames_per_proc, self.num_procs)
        self.total_rewards= torch.zeros(*shape, device=self.device)
        self.pretraining= pretraining #if set to True then do not use extrinsic rewards
        #initialize intrinsic rewards
        self.intrinsic_rewards=torch.zeros(*shape, device=self.device)
        # add monitorization for intrinsic part
        self.log_episode_return_int = torch.zeros(self.num_procs, device=self.device)
        self.log_return_int =  [0] * self.num_procs

        self.skills = self.diayn_reward.sample_skills(self.num_procs).tolist()
        
            # One-hot encode skills for each environment to be fed to the actor-critic networks
        self.one_hot_skills = self.diayn_reward.hot_encode_skills(torch.tensor(self.skills,device=self.device))
        self.skills_tracker = torch.zeros(*shape, device = self.device)
  
    
    def collect_experiences(self):
        """Collects rollouts and computes advantages.

        Runs several environments concurrently. The next actions are computed
        in a batch mode for all environments at the same time. The rollouts
        and advantages from all environments are concatenated together.

        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_frames_per_proc * num_envs, ...). k-th block
            of consecutive `self.num_frames_per_proc` frames contains
            data obtained from the k-th environment. Be careful not to mix
            data from different environments!
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.
        """
        
    
        for i in range(self.num_frames_per_proc):
            self.total_frames+=self.num_procs
            # Do one agent-environment interaction
            preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)

            with torch.no_grad():
                if self.acmodel.recurrent:
                    dist, value, memory = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1),self.one_hot_skills)
                else:
                    dist, value = self.acmodel(preprocessed_obs,skill=self.one_hot_skills)
           
            entropy = dist.entropy().detach() #I added detach here
          
            action = dist.sample()
            #print('action',action)
            if self.vizualise_video==True:
                self.frames.append(np.moveaxis(self.env.envs[0].get_frame(), 2, 0))
            
            obs, reward, terminated, truncated, agent_loc, _ = self.env.step(action.cpu().numpy())
            for agent_state in agent_loc:
                if agent_state in self.state_visitation_pos.keys():
                    self.state_visitation_pos[agent_state] += 1
                else:
                    self.state_visitation_pos[agent_state] = 1          
            # obs, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy())
            #print('the reward is', reward)
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
                  
            done = tuple(a | b for a, b in zip(terminated, truncated))

            for p in range(self.num_procs):
                obs_tuple=tuple( obs[p]['image'].reshape(-1).tolist())
                #print('obs tuple',len(obs_tuple))
                if obs_tuple in self.train_state_count:
                    self.train_state_count[obs_tuple]+= 1
                else:
                    self.train_state_count[obs_tuple]=1
         
           
            self.obss[i] = self.obs
            self.obs = obs #this is the next state
            if self.acmodel.recurrent:
                self.memories[i] = self.memory
                self.memory = memory
            self.masks[i] = self.mask  
            self.mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
            self.actions[i] = action
            #print('action',action)
            self.values[i] = value
            if self.reshape_reward is not None:
                self.rewards[i] = torch.tensor([
                    self.reshape_reward(obs_, action_, reward_, done_)
                    for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
                ], device=self.device)
            else:
                self.rewards[i] = torch.tensor(reward, device=self.device)
                ##print('self.rewards[i]',self.rewards[i])
            
            preprocessed_next_obs = self.preprocess_obss(self.obs, device=self.device)

            ###
            
            diayn_rewards = []

                # Gets the probability that the discriminator gives to the correct skill
            discriminator_predicted_probabilities_unnormalised = self.diayn_discriminator(preprocessed_next_obs)
            probability_of_correct_skill = F.softmax(discriminator_predicted_probabilities_unnormalised)

            for idx,ob in enumerate(preprocessed_next_obs.image):
                diayn_reward = self.diayn_reward.diayn_reward(probability_of_correct_skill[idx,self.skills[idx]])
                diayn_rewards.append(diayn_reward)
            
            # Add the intrinsic reward to the the extrinsic/envs reward
            total_reward = torch.tensor(reward, dtype=torch.float32,device='cuda')  # Ensure reward is float and requires grad
            diayn_rewards = torch.tensor(diayn_rewards, dtype=torch.float32, requires_grad=False)
            self.intrinsic_rewards[i]= diayn_rewards.clone().detach()
            # print('self.intrinsic_rewards[i]',self.intrinsic_rewards[i].requires_grad)
            # print('total_reward',total_reward.requires_grad)
            self.total_rewards[i] = total_reward.clone() + self.intrinsic_reward_coeff* self.intrinsic_rewards[i]
            

            
            if self.pretraining != 'False':
                # print('using intrinsic rewards only')
                self.total_rewards[i]= self.intrinsic_reward_coeff * self.intrinsic_rewards[i]

            self.intrinsic_reward_per_frame=(torch.mean(self.intrinsic_reward_coeff * self.intrinsic_rewards[i])).item()
            temp_rewards_int=self.intrinsic_reward_coeff*self.intrinsic_rewards[i]
            if self.singleton_env != 'False':
                # print('yes singleton intrinsic rewards')
                
                for idx in range(len( temp_rewards_int)):
                    #print(self.intrinsic_rewards[i])
                    if agent_loc[idx] in self.ir_dict.keys():
                        #print(self.intrinsic_rewards[i][idx])
                        self.ir_dict[agent_loc[idx]] +=  temp_rewards_int[idx].item()
                    else:
                        self.ir_dict[agent_loc[idx]] =  temp_rewards_int[idx].item()

            self.skills_tracker[i] = torch.tensor(self.skills, device=self.device)
            self.log_probs[i] = dist.log_prob(action)

            # Update log values

            self.log_episode_return += torch.tensor(reward, device=self.device, dtype=torch.float)
            #print(" self.log_episode_return", self.log_episode_return)
            self.log_episode_reshaped_return += self.rewards[i]
            self.log_episode_num_frames += torch.ones(self.num_procs, device=self.device)
            #print(" self.log_episode_num_frames", self.log_episode_num_frames)
            self.log_episode_return_int += self.intrinsic_rewards[i].clone().detach()
            #print('self.log_episode_return_int',self.log_episode_return_int)
            for i, done_ in enumerate(done): #for any done episode in any process we append it to log_return
                if done_:
                    #print("i",i) 
                    #print('done',done_)
                    self.log_done_counter += 1
                    self.log_return.append(self.log_episode_return[i].item())

                    #print(" self.log_return", self.log_return)
                    self.log_reshaped_return.append(self.log_episode_reshaped_return[i].item())
                    self.log_num_frames.append(self.log_episode_num_frames[i].item())
                    self.log_return_int.append(self.log_episode_return_int[i].item())
                    self.skills[i] = self.diayn_reward.sample_skills(1).item()
                    self.one_hot_skills = self.diayn_reward.hot_encode_skills(torch.tensor(self.skills,device=self.device))
                    

            self.log_episode_return *= self.mask #to reset when the episode is done
            self.log_episode_reshaped_return *= self.mask
            self.log_episode_num_frames *= self.mask
            self.log_episode_return_int *= self.mask
        
        ## store your skills    
        exps = DictList()
        
        # Add advantage and return to experiences

        preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
        with torch.no_grad():
            if self.acmodel.recurrent:
                _, next_value, _ = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1),self.one_hot_skills)
            else:
                _, next_value = self.acmodel(preprocessed_obs,self.one_hot_skills)

        for i in reversed(range(self.num_frames_per_proc)):
            #print('i',i)
            next_mask = self.masks[i+1] if i < self.num_frames_per_proc - 1 else self.mask
            #print('next_mask',next_mask)
            next_value = self.values[i+1] if i < self.num_frames_per_proc - 1 else next_value
            next_advantage = self.advantages[i+1] if i < self.num_frames_per_proc - 1 else 0
            #print('next_advantages',next_advantage)

            delta = self.total_rewards[i] + self.discount * next_value * next_mask - self.values[i]
            #print('delta',delta)
            self.advantages[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask
            #print("self.advantages[i]",self.advantages[i])
            #print("self.gae",self.gae_lambda)

        # Define experiences:
        #   the whole experience is the concatenation of the experience
        #   of each process.
        # In comments below:
        #   - T is self.num_frames_per_proc,
        #   - P is self.num_procs,
        #   - D is the dimensionality.

        
        exps.obs = [self.obss[i][j]
                    for j in range(self.num_procs)
                    for i in range(self.num_frames_per_proc)]
        if self.acmodel.recurrent:
            # T x P x D -> P x T x D -> (P * T) x D
            exps.memory = self.memories.transpose(0, 1).reshape(-1, *self.memories.shape[2:])
            # T x P -> P x T -> (P * T) x 1
            exps.mask = self.masks.transpose(0, 1).reshape(-1).unsqueeze(1)
        # for all tensors below, T x P -> P x T -> P * T
        #print("self.actions",self.actions)
        exps.action = self.actions.transpose(0, 1).reshape(-1)
        #print("self.actions",exps.action)
        exps.value = self.values.transpose(0, 1).reshape(-1)
        exps.reward = self.rewards.transpose(0, 1).reshape(-1)
        exps.advantage = self.advantages.transpose(0, 1).reshape(-1)
        exps.returnn = exps.value + exps.advantage
        exps.log_prob = self.log_probs.transpose(0, 1).reshape(-1)
        exps.skill = self.skills_tracker.transpose(0, 1).reshape(-1)
        #add exps.skills
      
        # Preprocess experiences

        exps.obs = self.preprocess_obss(exps.obs, device=self.device)

        # Log some values

        keep = max(self.log_done_counter, self.num_procs)
        #print("self.log_done_counter",self.log_done_counter)f
        #print("self.log_return",self.log_return)
        self.number_of_visited_states= len(self.train_state_count)
         #size of the grid and the possible combinations of object index, color and status
        self.state_coverage= self.number_of_visited_states
        #self.state_coverage_position=len(self.state_visitation_pos)
        non_zero_count=0
        for key, value in self.state_visitation_pos.items():
            if value != 0:
                non_zero_count += 1
        self.state_coverage_position= non_zero_count

        logs = {
            "return_per_episode": self.log_return[-keep:], #u keep the log of the last #processes episode returns
            "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "num_frames": self.num_frames,
            "return_int_per_episode": self.log_return_int[-keep:],
            "state_coverage": self.state_coverage,
            "frame_first_reward": self.saved_frame_first_reward,
            "frame_second_reward": self.saved_frame_second_reward,
            "frame_third_reward": self.saved_frame_third_reward,
            "state_visitation_pos": self.state_visitation_pos,
            "state_coverage_position":self.state_coverage_position,
            "reward_int_per_frame":self.intrinsic_reward_per_frame,
            "ir_dict":self.ir_dict
        }
        #print("self.log_return[-keep:]",self.log_return[-keep:])

        self.log_done_counter = 0
        self.log_return_int = self.log_return_int[-self.num_procs:]
        self.log_return = self.log_return[-self.num_procs:]
        #print('self.log_return',self.log_return)
        self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
        self.log_num_frames = self.log_num_frames[-self.num_procs:]

        return exps, logs, self.frames
    
    def update_parameters(self, exps):
        kl_div= self._calculate_KL_Div(exps)
        # Collect experiences
        for _ in range(self.epochs):
            # Initialize log values

            log_entropies = []
            log_values = []
            log_policy_losses = []
            log_value_losses = []
            log_grad_norms = []
            log_discriminator_losses=[]
            # log_batch_diversity=[]

            for inds in self._get_batches_starting_indexes():
                # Initialize batch values
                #print('indexes',inds)
                #print('len',len(inds))
                batch_entropy = 0
                batch_value = 0
                batch_policy_loss = 0
                batch_value_loss = 0
                batch_loss = 0
                batch_discriminator_loss = 0
                #batch_diversity = 0
                # Initialize memory

                if self.acmodel.recurrent:
                    memory = exps.memory[inds]

                for i in range(self.recurrence):
                    # Create a sub-batch of experience
                   
                    sb = exps[inds + i]
                    update_skills = self.diayn_reward.hot_encode_skills(sb.skill)

                    # Compute loss

                    if self.acmodel.recurrent:
                        dist, value, memory = self.acmodel(sb.obs, memory * sb.mask,skill=update_skills)
                    else:
                        dist, value = self.acmodel(sb.obs,skill=update_skills)
                    #print('sb.skills',sb.skills)
                    
                    entropy = dist.entropy().mean()

                    ratio = torch.exp(dist.log_prob(sb.action) - sb.log_prob)
                    surr1 = ratio * sb.advantage
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb.advantage
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_clipped = sb.value + torch.clamp(value - sb.value, -self.clip_eps, self.clip_eps)
                    surr1 = (value - sb.returnn).pow(2)
                    surr2 = (value_clipped - sb.returnn).pow(2)
                    value_loss = torch.max(surr1, surr2).mean()

                    loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss
                    discriminator_outputs = self.diayn_discriminator(sb.obs)
                    skills = sb.skill.long()
                    discriminator_loss = nn.CrossEntropyLoss()(discriminator_outputs, skills)

                    
                    # Update batch values

                    batch_entropy += entropy.item()
                    batch_value += value.mean().item()
                    batch_policy_loss += policy_loss.item()
                    batch_value_loss += value_loss.item()
                    batch_loss += loss
                    batch_discriminator_loss+=discriminator_loss
                 

                    # Update memories for next epoch

                    if self.acmodel.recurrent and i < self.recurrence - 1:
                        exps.memory[inds + i + 1] = memory.detach()

                # Update batch values

                batch_entropy /= self.recurrence
                batch_value /= self.recurrence
                batch_policy_loss /= self.recurrence
                batch_value_loss /= self.recurrence
                batch_loss /= self.recurrence
                batch_discriminator_loss/=self.recurrence
                # Update actor-critic

                self.optimizer.zero_grad()
                loss = batch_loss  + batch_discriminator_loss 
                loss.backward()

                #global_grad_norm_(list(self.acmodel.parameters())+list(self.diayn_discriminator.parameters()))
                grad_norm = sum(p.grad.data.norm(2) ** 2 for p in self.acmodel.parameters()) ** 0.5
               
                torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
               
                # Update log values

                log_entropies.append(batch_entropy)
                log_values.append(batch_value)
                log_policy_losses.append(batch_policy_loss)
                log_value_losses.append(batch_value_loss)
                log_grad_norms.append(grad_norm.item())
                # print('log_grad_norms',log_grad_norms)
                log_discriminator_losses.append(batch_discriminator_loss.item())
               
                #print('log_policy_losses',log_policy_losses)
                #print('log_discriminator_losses',log_discriminator_losses)

        # Log some values
           
        logs = {
            "entropy": np.mean(log_entropies),
            "value": np.mean(log_values),
            "policy_loss": np.mean(log_policy_losses),
            "value_loss": np.mean(log_value_losses),
            "grad_norm": np.mean(log_grad_norms),
            "discriminator_loss": np.mean(log_discriminator_losses),
            "KL_div":kl_div
        }

        return logs
        
    def _calculate_KL_Div(self,exps): #Taking all num_frames experiences collected to calculate div
        # indexes = np.arange(0, self.num_frames, self.recurrence)
        # indexes = np.random.permutation(indexes)
        # random_batch_indexes = [indexes[0:self.num_frames]]
        # print('random_batch_indexes',random_batch_indexes)
        # sb = exps[random_batch_indexes]
        sb=exps
        memory = exps.memory
        prob_dist_per_skill= {}
        for i in range (self.no_skills):
            skill_one_hot = torch.zeros(self.no_skills)
            skill_one_hot[i]=1
            #print('sb.obs',sb.obs.image.shape)
            skill_all= np.tile(skill_one_hot,(sb.obs.image.shape[0],1))
            #print('skill all',skill_all)
            skill_all_tensor= torch.tensor(skill_all, dtype=torch.int32, device=self.device)
            
            if self.acmodel.recurrent:
                prob_dist,_,_ = self.acmodel(sb.obs, memory * sb.mask,skill=skill_all_tensor)
            else:
                prob_dist,_ = self.acmodel(sb.obs,skill=skill_all_tensor)

            #print('prob_dist probabilities',prob_dist.probs)
            prob_dist_per_skill[i]= prob_dist.probs
        #print('prob dist per skill',prob_dist_per_skill)

        #calculate KL div
       

# Calculate KL divergence between all pairs of probability tensors
        
        kl_divergences = torch.zeros((self.no_skills, self.no_skills))

        for i in range(self.no_skills):
            for j in range(self.no_skills):
                kl_divergences[i, j] = kl_divergence(prob_dist_per_skill[i], prob_dist_per_skill[j])

        #print("KL Divergence between the probability distributions:")
        #print(kl_divergences)
        total_combinations = self.no_skills * (self.no_skills - 1) 

        # Sum up all the individual KL divergences (excluding diagonals)
        average_kl_divergence = kl_divergences.sum()/total_combinations

    

        #print('hey',average_kl_divergence.item())
        return average_kl_divergence.item()

def kl_divergence(p, q):
    #print(p.log())
    return torch.nn.functional.kl_div(p.log(), q, reduction='batchmean')


        

    
