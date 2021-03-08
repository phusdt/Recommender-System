import torch

# create the Restricted Boltzmann Machine architecture
class RBM:
    def __init__(self, n_v, n_h):
        """
        Initalize the parameters (weights and biases) we optimize during the training the training process
        param: 
            n_v: number of visible units
            v_h: number of hidden units
        """

        # Weight used for the probability of the visible units given the hidden units
        self.W = torch.nn.init.xavier_normal_(n_h, n_v) # random normal distribution mean = 0, variance = 1

        # Bias probability of the hidden units is actived, given the values of
        # the visible units ( p_h_given_v )
        self.h_bias = torch.nn.init.xavier_normal_(1, n_h)

        # Bias probability of the visible units is activated, given the value of the
        # hidden units ( p_v_given_h )
        self.v_bias = torch.nn.init.xavier_normal_(1, n_v)
    
    def sample_h(self, x):
        """
        Sample the hidden units
        param:
            x: dataset
        """

        # Probability v is activated given that the value h is sigmoid( Wx + a )
        wx = torch.mm(x, self.W.t())

        # Expand the mini-batch
        activation = wx + self.h_bias.expand_as(wx)

        # Calculate the probability p_h_given_v
        p_h_given_v = torch.sigmoid(activation)

        # Construct a Bernoulli RBM to predict whether an user loves the movie or not ( 0 or 1 )
        # This corresponds to whether the v_h is actived or not actived 
        return  p_h_given_v, torch.bernoulli(p_h_given_v)

    def sample_v(self, y):
        """
        Sample the visible units
        param:
            y: the dataset
        """

        # Probability v is activated given that the value h is sigmoid( Wx + a )
        wy = torch.mm(y, self.W)

        # Expand the mini-batch
        activation = wy + self.v_bias.expand_as(wy)

        # Calculate the probability p_v_given_h
        p_v_given_h = torch.sigmoid(activation)

        # Construct a Bernoulli RBM to predict whether an user loves the movie or not ( 0  or 1 )
        # This corresponds to whether the v_h is actived or not actived
        return p_v_given_h, torch.bernoulli(p_v_given_h)

    def train(self, v0, vk, ph0, phk):
        """
        Perform contrastive divergence algorithm to optimize the weights that minimize the energy
        This maximize the log-likelihood of model
        """

        # Approximate the gradients with the CD algorithm
        self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()

        # Add (difference, 0) for the tensor of 2 dimensions
        self.v_bias = torch.sum((v0 - vk), 0)
        self.h_bias = torch.sum((ph0 - phk), 0)

