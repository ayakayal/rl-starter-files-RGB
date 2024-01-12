from cts_model import ConvolutionalMarginalDensityModel, ConvolutionalDensityModel, L_shaped_context, LocationDependentDensityModel,dilations_context
import torch.nn.functional as F
import cv2
import numpy as np
import torch
import scipy
from scipy import misc
from PIL import Image
FRSIZE = 28 #IT WAS 56
MAXVAL = 255                # original max value for a state
MAX_DOWNSAMPLED_VAL = None   # downsampled max value for a state. 8 in the paper.
# FEATURE_NUM = 512
import math
from collections import defaultdict
class PC():
    # class for process with pseudo count rewards
    def __init__(self):
        # initialize
        global MAX_DOWNSAMPLED_VAL
        self.count_coeff=0.01
       
        MAX_DOWNSAMPLED_VAL = 8
        #self.CTS = ConvolutionalMarginalDensityModel((FRSIZE, FRSIZE))            # 100 iter/s for memory filling
        #self.CTS = ConvolutionalDensityModel((FRSIZE, FRSIZE), L_shaped_context) # 12 iter/s for memory filling
        
        self.CTS = LocationDependentDensityModel((FRSIZE, FRSIZE),dilations_context) # 12 iter/s
       
        # print ("Downsampled to " + str(MAX_DOWNSAMPLED_VAL))
        # self.flat_pixel_counter = np.zeros((FRSIZE*FRSIZE, MAX_DOWNSAMPLED_VAL+1)) # Counter for each (pos1, pos2, val), used for joint method
        # self.flat_feature_counter = np.zeros((FEATURE_NUM, MAX_DOWNSAMPLED_VAL + 1))
        self.total_num_states = 0  # total number of seen states

        self.n = 0
        # self.dict = defaultdict(int)

    # def clear(self):
    #     self.flat_pixel_counter = np.zeros((FRSIZE*FRSIZE, MAX_DOWNSAMPLED_VAL+1))
    #     self.total_num_states = 0
    #     return


    def pc_reward(self, state):
        """
        The final API used by others.
        Given an state, return back the final pseudo count reward
        :return:
        """
        state = self.preprocess(state) #skipped the preprocessing step because it was changing shape from (56,56,3) to (56,56)
        pc_reward = self.add(state)

        return pc_reward



    # def pc_reward_feature(self, feature):
    #     # scale feature to 0 - 1
    #     feature = self.standardize_feature(feature)
    #     # discretize features
    #     feature = self.discretize_feature(feature)
    #     print feature
    #     if self.method == 'joint':
    #         # Model each pixel as independent pixels.
    #         # p = (c1/n) * (c2/n) * ... * (cn/n)
    #         # pp = (c1+1)/(n+1) * (c2+1)/(n+1) ...
    #         # N = (p/pp * (1-pp))/(1-p/pp) ~= (p/pp) / (1-p/pp)
    #         state = feature
    #         if self.total_num_states > 0:
    #             nr = (self.total_num_states + 1.0) / self.total_num_states
    #             pixel_count = self.flat_feature_counter[range(FEATURE_NUM), state]
    #             self.flat_feature_counter[range(FEATURE_NUM), state] += 1
    #             p_over_pp = np.prod(nr * pixel_count / (1.0 + pixel_count))
    #             denominator = 1.0 - p_over_pp
    #             if denominator <= 0.0:
    #                 print "psc_add_image: dominator <= 0.0 : dominator=", denominator
    #                 denominator = 1.0e-20
    #             pc_count = p_over_pp / denominator
    #             pc_reward = self.count2reward(pc_count)
    #         else:
    #             pc_count = 0.0
    #             pc_reward = self.count2reward(pc_count)
    #         self.total_num_states += 1
    #         return pc_reward

    # def standardize_feature(self, feature):
    #     min_feature = np.min(feature)
    #     max_feature = np.max(feature)
    #     feature = (feature - min_feature) / (max_feature - min_feature)
    #     return feature

    # def discretize_feature(self, feature):
    #     feature = np.asarray(MAX_DOWNSAMPLED_VAL * feature, dtype=int)
    #     return feature




    # def preprocess(self, state):
    #     #print('before preproc',state.shape)
    #     state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY) # transform to gray scale
    #     #print('after gray',state.shape)
    #     state = cv2.resize(state, (FRSIZE, FRSIZE)) #just scaling
    #     #print('after resize',state.shape)
    #     state = np.uint8(state / MAXVAL * MAX_DOWNSAMPLED_VAL) #i commented out normalization
    #     state= torch.tensor(state, device='cuda', dtype=torch.uint8)
    #     return state

    def preprocess(self, state):
    
        frame = Image.fromarray(state)

        # Specify the new dimensions
        new_height, new_width = (FRSIZE,FRSIZE)

        # Resize the image using bilinear interpolation
        resized_frame = frame.resize((new_width, new_height), Image.BILINEAR)
        # Convert the result back to a NumPy array if needed
        resized_frame = np.array(resized_frame)
        # print(resized_frame.size)
        state= torch.tensor(resized_frame, device='cuda', dtype=torch.uint8) 
        # print(state.shape)
        return state
    

    def add(self, state):
        self.n += 1
       
            # Model described in the paper "Unifying Count-Based Exploration and Intrinsic Motivation"
        log_p = self.CTS.update(state)
        #print('log_p',log_p)
        log_pp = self.CTS.query(state)
        #print('log_pp',log_pp)
        n = self.p_pp_to_count(log_p, log_pp)
        pc_reward = self.count2reward(n)
        #print('pc_reward',pc_reward)
        ## Following codes are used for generating images during training for debug
        # if self.n == 200:
        #     import matplotlib.pyplot as plt
        #     img = self.CTS.sample()
        #     plt.imshow(img)
        #     plt.show()
        return pc_reward

    #
    def p_pp_to_count(self, log_p, log_pp):
        """
        :param p: density estimation p. p = p(x;x_{<t})
        :param pp: recording probability. p' = p(x;x_{<t}x)
        :return: N = p(1-pp)/(pp-p) = (1-pp)/(pp/p-1) ~= 1/(pp/p-1)
        pp/p = e^(log_pp) / e^(log_p) = e ^ (log_pp - log_p)
        """
        #assert log_pp >= log_p
        # prediction_gain= F.relu(log_pp - log_p)
        # N= 1. / (np.exp(self.count_coeff / math.sqrt(self.n) *prediction_gain) - 1.0)
        
        #pp = np.exp(log_pp)
        PG=log_pp - log_p
        #assert pp <= 1
        if PG<=0:
            N=0
        else:
            pp_over_p = np.exp(PG)
            #N = (1.0-pp) / (pp_over_p - 1)
            N= 1/(pp_over_p-1)
        return N   #u have to fix this


    def count2reward(self, count, alpha=0.01, power=-0.5):
        # r = beta (N + alpha)^power
        # print('count',count)
        reward = ((count+alpha) ** power)
        
        return reward