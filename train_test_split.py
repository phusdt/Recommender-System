from __future__ import print_function
import numpy as np
import pandas as pd
from pandas import ExcelFile
from scipy import sparse
import timeit
start = timeit.default_timer()
from sklearn.metrics import mean_squared_error
from math import sqrt

class RBM:
  def __init__(self, num_visible, num_hidden):
    self.num_hidden = num_hidden
    self.num_visible = num_visible
    self.debug_print = True

    # Initialize a weight matrix, of dimensions (num_visible x num_hidden), using
    # a uniform distribution between -sqrt(6. / (num_hidden + num_visible))
    # and sqrt(6. / (num_hidden + num_visible)). One could vary the
    # standard deviation by multiplying the interval with appropriate value.
    # Here we initialize the weights with mean 0 and standard deviation 0.1.
    # Reference: Understanding the difficulty of training deep feedforward
    # neural networks by Xavier Glorot and Yoshua Bengio
    np_rng = np.random.RandomState(1234)

    self.weights = np.asarray(np_rng.uniform(
         low=-0.1 * np.sqrt(6. / (num_hidden + num_visible)),
            high=0.1 * np.sqrt(6. / (num_hidden + num_visible)),
                           size=(num_visible, num_hidden)))

    # Tính trọng số cho đơn vị sai số cho dòng 1 cột 1
    # Insert weights for the bias units into the first row and first column.
    self.weights = np.insert(self.weights, 0, 0, axis = 0)
    self.weights = np.insert(self.weights, 0, 0, axis = 1)

  def train(self, data, max_epochs = 5000, learning_rate = 0.1):
    """
    Train the machine.

    Parameters
    ----------
    data: A matrix where each row is a training example consisting of the states of visible units.
    """
    num_examples = data.shape[0]
    # Insert bias units of 1 into the first column.
    data = np.insert(data, 0, 1, axis = 1)

    for epoch in range(max_epochs):
      # Clamp to the data and sample from the hidden units.
      # (This is the "positive CD phase", aka the reality phase.)
      pos_hidden_activations = np.dot(data, self.weights)
      pos_hidden_probs = self._logistic(pos_hidden_activations)
      pos_hidden_probs[:,0] = 1 # Fix the bias unit.
      pos_hidden_states = pos_hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)
      # Note that we're using the activation *probabilities* of the hidden states, not the hidden states
      # themselves, when computing associations. We could also use the states; see section 3 of Hinton's
      # "A Practical Guide to Training Restricted Boltzmann Machines" for more.
      pos_associations = np.dot(data.T, pos_hidden_probs)
      # Reconstruct the visible units and sample again from the hidden units.
      # (This is the "negative CD phase", aka the daydreaming phase.)
      neg_visible_activations = np.dot(pos_hidden_states, self.weights.T)
      neg_visible_probs = self._logistic(neg_visible_activations)
      neg_visible_probs[:,0] = 1 # Fix the bias unit.
      neg_hidden_activations = np.dot(neg_visible_probs, self.weights)
      neg_hidden_probs = self._logistic(neg_hidden_activations)
      # Note, again, that we're using the activation *probabilities* when computing associations, not the states
      # themselves.
      neg_associations = np.dot(neg_visible_probs.T, neg_hidden_probs)
      # Kích hoạt trọng số
      # Update weights.
      self.weights += learning_rate * ((pos_associations - neg_associations) / num_examples)
      error = np.sum((data - neg_visible_probs) ** 2)
      #if self.debug_print:
      print('epoch:')
      print("Epoch %s: error is %s" % (epoch, error))

  def run_visible(self, data):
    """
    Assuming the RBM has been trained (so that weights for the network have been learned),
    run the network on a set of visible units, to get a sample of the hidden units.

    Parameters
    ----------
    data: A matrix where each row consists of the states of the visible units.

    Returns
    -------
    hidden_states: A matrix where each row consists of the hidden units activated from the visible
    units in the data matrix passed in.
    """

    num_examples = data.shape[0]

    # Create a matrix, where each row is to be the hidden units (plus a bias unit)
    # sampled from a training example.
    hidden_states = np.ones((num_examples, self.num_hidden + 1))
    # Insert bias of 1 into the first clumn of data
    data = np.insert(data, 0, 1, axis=1)
    # Tính hàm kích hoạt cho đơn vị ẩn
    # Calculate the activations of the hidden units.
    hidden_activations = np.dot(data, self.weights)
    # Calculate the probabilities of turning the hidden units on. # Tính hàm khả năng chuyển trạng thái của đơn vị ẩn
    hidden_probs = self._logistic(hidden_activations)
    # Turn the hidden units on with their specified probabilities.
    hidden_states[:,:] = hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)
    # Always fix the bias unit to 1.
    # hidden_states[:,0] = 1 examples, self.num_hidden + 1))
    # Ignore the bias units.
    hidden_states = hidden_states[:,1:]
    return hidden_states

  # TODO: Remove the code duplication between this method and `run_visible`?
  def run_hidden(self, data):
    """
    Assuming the RBM has been trained (so that weights for the network have been learned),
    run the network on a set of hidden units, to get a sample of the visible units.

    Parameters
    ----------
    data: A matrix where each row consists of the states of the hidden units.

    Returns
    -------
    visible_states: A matrix where each row consists of the visible units activated from the hidden
    units in the data matrix passed in.
    """

    num_examples = data.shape[0]

    # Create a matrix, where each row is to be the visible units (plus a bias unit)
    # sampled from a training example.
    visible_states = np.ones((num_examples, self.num_visible + 1))

    # Insert bias units of 1 into the first column of data.
    data = np.insert(data, 0, 1, axis = 1)

    # Calculate the activations of the visible units.
    visible_activations = np.dot(data, self.weights.T)
    # Calculate the probabilities of turning the visible units on.
    visible_probs = self._logistic(visible_activations)
    # Turn the visible units on with their specified probabilities.
    visible_states[:,:] = visible_probs > np.random.rand(num_examples, self.num_visible + 1)
    # Always fix the bias unit to 1.
    visible_states[:,0] = 1

    # Ignore the bias units.
    visible_states = visible_states[:,1:]
    return visible_states


  def daydream(self, num_samples):
    """
    Randomly initialize the visible units once, and start running alternating Gibbs sampling steps
    (where each step consists of updating all the hidden units, and then updating all of the visible units),
    taking a sample of the visible units at each step.
    Note that we only initialize the network *once*, so these samples are correlated.

    Returns
    -------
    samples: A matrix, where each row is a sample of the visible units produced while the network was
    daydreaming.
    """

    # Create a matrix, where each row is to be a sample of of the visible units
    # (with an extra bias unit), initialized to all ones.
    samples = np.ones((num_samples, self.num_visible + 1))

    # Take the first sample from a uniform distribution.
    samples[0,1:] = np.random.rand(self.num_visible)

    # Start the alternating Gibbs sampling.
    # Note that we keep the hidden units binary states, but leave the
    # visible units as real probabilities. See section 3 of Hinton's
    # "A Practical Guide to Training Restricted Boltzmann Machines"
    # for more on why.
    for i in range(1, num_samples):
      visible = samples[i-1,:]
      # Calculate the activations of the hidden units.
      hidden_activations = np.dot(visible, self.weights)
      # Calculate the probabilities of turning the hidden units on.
      hidden_probs = self._logistic(hidden_activations)
      # Turn the hidden units on with their specified probabilities.
      hidden_states = hidden_probs > np.random.rand(self.num_hidden + 1)
      # Always fix the bias unit to 1.
      hidden_states[0] = 1
      # Recalculate the probabilities that the visible units are on.
      visible_activations = np.dot(hidden_states, self.weights.T)
      visible_probs = self._logistic(visible_activations)
      visible_states = visible_probs > np.random.rand(self.num_visible + 1)
      samples[i,:] = visible_states

    # Ignore the bias units (the first column), since they're always set to 1.
    return samples[:,1:]

  def _logistic(self, x):
    return 1.0 / (1 + np.exp(-x))


# Ham softmax va softmax_ stable -tru moi cot cua ma tran dau vao Z di 1 gia tri lon nhat trong cot do. Ta co phien ban on dinh hon
def softmax(X):
  X_exp = np.exp(X)
  partition = X_exp.sum(axis=1, keepdims=True)
  return X_exp / partition  # The broadcast mechanism is applied here

# def softmax_stable(Z):
#     """
#     Compute softmax values for each sets of scores in Z.
#     each column of Z is a set of score.
#     """
#     e_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
#     A = e_Z / e_Z.sum(axis=0)
#     return A
# Ham du doan class cho dl moi


#######################
if __name__ == '__main__':
  r = RBM(num_visible= 1712, num_hidden=19)
  Train = pd.read_csv(r'C:\Users\PC\Desktop\merge.csv')
  Test = pd.read_csv(r'C:\Users\PC\Desktop\testing_data.csv')
  # print('Tap train :')
  # print(Train)
  # print('Tap test :')
  # print(Test)
########################
  # Tap du dau vao de train
  training_data = np.array(Train)

  print('Tap train :')
  print(training_data)
  # Chay dl dau vao
  # print('So dong:', testing_data.shape[0])
  # print('So cot:', testing_data.shape[1])
  def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

  # rmse_val = rmse(np.array(d), np.array(p))

  r.train(training_data, max_epochs = 5000)
  print ('Weight:')
  print(r.weights)
  # print('Chieu dai weight:', len(r.weights)) 1713

  X = (r.weights)
  X_prob = softmax(X)
  print('weight sau khi qua softmax:')
  print(X_prob, X_prob.sum(axis=1))

  # X_prob = softmax(X)
  # print('train sau khi qua softmax:')
  # print(X_prob, X_prob.sum(axis=1))

  # A = (r.weights)
  # A_prob = softmax_stable(A)
  # print('Weight sau khi qua softmax_stable:')
  # print(A_prob, A.sum(axis=0, keepdims=True), '\n',A_prob.sum(axis=1))

################################### Test
  testing_data = np.array(Test.head(5))
  print("Trang thai cac don vi an tu don vi hien thi _ Testing: ")
  print(r.run_visible(testing_data))
  print('So dong',(r.run_visible(testing_data).shape[0]))
  print('So cot',(r.run_visible(testing_data).shape[1]))

  df = pd.DataFrame(r.run_visible(testing_data))
  df.to_csv(r'C:\Users\PC\Desktop\testing.csv', index=False, header=True)

########################################################

# Time
stop = timeit.default_timer()
print('Time: ', stop - start, "second")
###################
