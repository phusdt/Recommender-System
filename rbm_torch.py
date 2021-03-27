import torch


# Create the Restricted Boltzmann Machine architecture
class RBM:
    def __init__(self, n_vis, n_hid, nb_epoch, nb_users, batch_size_):
        """
        Initialize the parameters (weights and biases) we optimize during the training process
        :param n_vis: number of visible units
        :param n_hid: number of hidden units
        """

        self.nb_epoch = nb_epoch
        self.nb_users = nb_users
        self.batch_size_ = batch_size_
        self.get_loss = []

        # Weights used for the probability of the visible units given the hidden units
        self.W = torch.randn(n_hid, n_vis)  # torch.rand: random normal distribution mean = 0, variance = 1

        # Bias probability of the visible units is activated, given the value of the hidden units (p_v_given_h)
        self.v_bias = torch.randn(1, n_vis)  # fake dimension for the batch = 1

        # Bias probability of the hidden units is activated, given the value of the visible units (p_h_given_v)
        self.h_bias = torch.randn(1, n_hid)  # fake dimension for the batch = 1

    def sample_h(self, x):
        """
        Sample the hidden units
        :param x: the dataset
        """

        # Probability h is activated given that the value v is sigmoid(Wx + a)
        # torch.mm make the product of 2 tensors
        # W.t() take the transpose because W is used for the p_v_given_h
        wx = torch.mm(x, self.W.t())

        # Expand the mini-batch
        activation = wx + self.h_bias.expand_as(wx)

        # Calculate the probability p_h_given_v
        p_h_given_v = torch.sigmoid(activation)

        # Construct a Bernoulli RBM to predict whether an user loves the movie or not (0 or 1)
        # This corresponds to whether the n_hid is activated or not activated
        return p_h_given_v, torch.bernoulli(p_h_given_v)

    def sample_v(self, y):
        """
        Sample the visible units
        :param y: the dataset
        """

        # Probability v is activated given that the value h is sigmoid(Wx + a)
        wy = torch.mm(y, self.W)

        # Expand the mini-batch
        activation = wy + self.v_bias.expand_as(wy)

        # Calculate the probability p_v_given_h
        p_v_given_h = torch.sigmoid(activation)

        # Construct a Bernoulli RBM to predict whether an user loves the movie or not (0 or 1)
        # This corresponds to whether the n_vis is activated or not activated
        return p_v_given_h, torch.bernoulli(p_v_given_h)

    def train(self, v0):
        """
        Perform contrastive divergence algorithm to optimize the weights that minimize the energy
        This maximizes the log-likelihood of the model
        """

        for epoch in range(1, self.nb_epoch + 1):
            train_recon_error = 0  # RMSE reconstruction error initialized to 0 at the beginning of training
            s = 0.  # a counter (float type)

            # Second for loop - go through every single user
            # Lower bound is 0, upper bound is (nb_users - batch_size_), batch_size_ is the step of each batch (512)
            # The 1st batch is for user with ID = 0 to user with ID = 511
            for id_user in range(0, self.nb_users - self.batch_size_, self.batch_size_):

                # At the beginning, v0 = vk. Then we update vk            
                v0, vk, ph0, phk = self._gibbs_sampling(v0[id_user:id_user + self.batch_size_])
                # Approximate the gradients with the CD algorithm
                self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()

                # Add (difference, 0) for the tensor of 2 dimensions
                self.v_bias = torch.sum((v0 - vk), 0)
                self.h_bias = torch.sum((ph0 - phk), 0)

                # Compare vk updated after the training to v0 (the target)
                train_recon_error += torch.sqrt(torch.mean((v0[v0 >= 0] - vk[v0 >= 0])**2))
                s += 1.

            # Update RMSE reconstruction error
            self.get_loss.append(train_recon_error / s)

            print('Epoch: ' + str(epoch) + '- RMSE Reconstruction Error: ' + str(train_recon_error.data.numpy() / s))

    def _gibbs_sampling(self, v0):
        """
        Gibbs distribution sampling
        """
        # number for the end condition of the while loop
        # At the beginning, v0 = vk. Then we update vk
        vk = v0
        ph0, _ = self.sample_h(v0)
        # 
        k = 10
        for i in range(k):
            _, hk = self.sample_h(vk)
            _, vk = self.sample_v(hk)

            # We don't want to learn when there is no rating by the user, and there is no update when rating = -1
            vk[v0 < 0] = v0[v0 < 0]
        phk, _ = self.sample_h(vk)
        
        return v0, vk, ph0, phk

    def predict(self, x):
        """
        Predict for user
        """
        
        _, h = self.sample_h(x)
        _, v = self.sample_v(h)

        return v
