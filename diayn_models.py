import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal
from torch.distributions.categorical import Categorical
import torch_ac
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        #print('oui')
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

def init_weight(layer, initializer="he normal"):
    if initializer == "xavier uniform":
        nn.init.xavier_uniform_(layer.weight)
    elif initializer == "he normal":
        nn.init.kaiming_normal_(layer.weight)


class Discriminator(nn.Module):
    def __init__(self, n_skills, n_hidden_filters=256):
        super(Discriminator, self).__init__()
        self.n_skills = n_skills
        self.n_hidden_filters = n_hidden_filters
        ###add the CNN to extract features from the image. Pytorch is using default paramters to initialize this
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
        ##############
        self.image_embedding_size= 40000 #it was 64 for non RGB
        self.hidden1 = nn.Linear(in_features=self.image_embedding_size, out_features=self.n_hidden_filters)
        init_weight(self.hidden1)
        self.hidden1.bias.data.zero_()
        self.hidden2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
        init_weight(self.hidden2)
        self.hidden2.bias.data.zero_()
        self.q = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_skills)
        init_weight(self.q, initializer="xavier uniform")
        self.q.bias.data.zero_()

    def forward(self, obs):
        x = obs.image.transpose(1, 3).transpose(2, 3)
        #print('x',x.shape)
        x = self.image_conv(x)
        embedding = x.reshape(x.shape[0], -1)
        x = F.relu(self.hidden1(embedding))
        x = F.relu(self.hidden2(x))
        unnormalized_probs = self.q(x)
        #pred_dist = Categorical(logits=F.log_softmax(unnormalized_probs, dim=1))
        return unnormalized_probs #was unnormalized probs
#V2_VERSION
class DIAYN_discriminator(nn.Module):
    def __init__(self, no_skills, use_memory=False, use_text=False):
        super().__init__()

        # Define image embedding
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
        n = 56
        m = 56
        self.image_embedding_size = 40000

        # Resize image embedding
        self.embedding_size = self.semi_memory_size
        print('self.embedding_size',self.embedding_size)

        # Define the discriminator network
        self.discriminator = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, no_skills)
        )        
        
        # Initialize parameters correctly
        self.apply(init_params)

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size
    
    def forward(self,next_state):
        """
        The forward takes as input the next_state and returns an unnormalized prob over skills latent space.
        """

        # apply the 3D CNN to the image 
        embedding = next_state.image.transpose(1, 3).transpose(2, 3)
        embedding = self.image_conv(embedding)
        embedding = embedding.reshape(embedding.shape[0], -1)

        # return an unnormalized distribution over the skills
        x = self.discriminator(embedding)

        return x

