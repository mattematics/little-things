class Net():
  """
  Implements a neural network with a user-determined number and size of hidden layers.
  Except for the final hidden layer, each hidden layer is followed
  by a non-linear function f(x) = x when x >= 0, x/100 when x < 0.
  Predicts by selecting the index of the maximum output.
  Trains by minibatch gradient descent using a softmax loss function + regularization.
  """
  def __init__(self, input_size, hidden_sizes, output_size, std=1e-4, bstd=1e-4):
    """
    Initializes the network.
    input_size = the length of each individual input vector
    hidden_sizes = a list specifying how many nodes are in each hidden layer, e.g. [], [2], [3,3]
    output_size = the number of class labels in the prediction task
    std = the standard deviation to use for initializing weights
    bstd = the standard deviation to use for initializing biases
    """
    num_hidden_layers = len(hidden_sizes)
    
    # initialize weight matrices
    self.weights = []
    if num_hidden_layers > 0:
        for i in xrange(num_hidden_layers):
            if i == 0:
                self.weights.append(std * np.random.randn(input_size, hidden_sizes[0]))
            else:
                self.weights.append(std * np.random.randn(hidden_sizes[i-1], hidden_sizes[i]))
        self.weights.append(std * np.random.randn(hidden_sizes[-1], output_size))
    else:
        self.weights.append(std * np.random.randn(input_size, output_size))
    
    # initialize bias vectors
    self.biases = []
    for i in xrange(num_hidden_layers):
        self.biases.append(bstd * np.random.randn(hidden_sizes[i]))
    self.biases.append(bstd * np.random.randn(output_size))
    
  def loss(self, X, y=None, reg=0.0):
    """
    Computes the class scores for X.
    X = numpy.array whose rows are the input vectors
    If y is provided, this function also computes the loss function
    as well as the gradients for use in training.
    y = numpy.array whose rows y[i] are the correct class labels (indices in 0,...,output_size-1) for the corresponding X[i]
    reg = the regularization parameter, which scales the regularization
    term in the loss function.
    """
    Ws = self.weights
    bs = self.biases
    N, D = X.shape # number of samples, number of features per sample

    # Compute the forward pass
    self.activations = []
    for i in xrange(len(Ws)): # for each set of weights
        W,b = Ws[i], bs[i]
        if i == 0:
            H = np.dot(X,W) + b
        else:
            H = np.dot(self.activations[-1],W) + b
        if i < len(Ws) - 1: # if we're computing hidden activations, apply nonlinear function
            H = (H > 0) * (H) + (H < 0) * (H/100.0)
        self.activations.append(H)
    scores = self.activations[-1]
    
    # If there's no labels provided, stop here
    if y is None:
      return scores

    # Compute the loss
    exped_scores = np.exp(scores)
    sums = np.sum(exped_scores,axis=1)
    # softmax classifier loss
    data_loss = (-1.0/N) * np.sum(np.log(exped_scores[range(N),y] / sums))

    # loss due to regularization
    reg_loss = 0
    for i in xrange(len(Ws)):
        reg_loss += np.sum(Ws[i]**2)
    reg_loss *= reg*(0.5)

    loss = data_loss + reg_loss
    
    # Compute gradients
    weights_grads = []
    biases_grads = []
    activation_grads = []
    for i in xrange(len(Ws)):
        weights_grads.append(np.copy(Ws[i]))
        biases_grads.append(np.copy(bs[i]))
        activation_grads.append(np.copy(self.activations[i]))

    DlossDscores = np.array(exped_scores / (N * np.matrix(sums).T))
    DlossDscores[range(N),y] -= (1.0/N)
    
    for i in xrange(len(Ws)-1,-1,-1):
        if i == 0:
            weights_grads[0] = np.dot(X.T, activation_grads[0]) + reg*Ws[0]
            biases_grads[0] = np.dot(np.ones((1,N)), activation_grads[0])[0]
        elif i == len(Ws)-1:
            H = self.activations[i-1]
            weights_grads[i] = np.dot(H.T, DlossDscores) + reg*Ws[i]
            biases_grads[i] = np.dot(np.ones((1,N)), DlossDscores)[0]
            dH = np.dot(DlossDscores, Ws[i].T)
            activation_grads[i-1] = dH
        else:
            H = self.activations[i-1]
            dH_out = activation_grads[i]
            weights_grads[i] = np.dot(H.T, dH_out) + reg*Ws[i]
            biases_grads[i] = np.dot(np.ones((1,N)), dH_out)[0]
            dH = np.dot(dH_out, Ws[i].T)
            dH = dH * (H > 0) + dH/100.0 * (H < 0)
            activation_grads[i-1] = dH
    
    grads = {}
    grads['weights'] = weights_grads
    grads['biases'] = biases_grads

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
    """
    Trains the network via minibatch gradient descent.
    X = numpy.array whose rows are the input vectors
    y = numpy.array whose rows y[i] are the correct class labels for X[i]
    learning_rate = number that scales the gradient descent steps
    learning_rate_decay = number that scales the learning_rate after each training epoch
    reg = the regularization parameter, scales the regularization term in the loss function
    num_iters = how many minibatch gradient descent steps to take
    batch_size = the size of each minibatch, if 0, will train with full batches
    verbose = bool, whether to print loss, learning rate during training
    """
    num_train = X.shape[0]
    if batch_size > 0:
      iterations_per_epoch = max(num_train / batch_size, 1)
    else:
      iterations_per_epoch = max(num_train / len(X), 1)

    for it in xrange(num_iters):
      X_batch = None
      y_batch = None

      # create minibatch
      if batch_size > 0:
        indices = np.random.choice(X.shape[0],batch_size)
        X_batch = X[indices]
        y_batch = y[indices]
      else:
        X_batch = X
        y_batch = y

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
    
      for i in xrange(len(self.weights)):
        self.weights[i] -= grads['weights'][i] * learning_rate
        self.biases[i] -= grads['biases'][i] * learning_rate

      if verbose and it % 100 == 0:
        print 'iteration %d / %d: loss %f, learning rate %f' % (it, num_iters, loss, learning_rate)

      # Every epoch, decay the learning rate
      if it % iterations_per_epoch == 0:
        learning_rate *= learning_rate_decay

  def predict(self, X):
    """
    Returns the predicted class labels (indices in 0,...,output_size) for input X.
    """
    y_pred = np.argmax(self.loss(X),axis=1)
    return y_pred