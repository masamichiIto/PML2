import numpy as np

class AdalineGD(object):
    """ ADAptive LIniera NEuron classifier
    
    parameters:
    * eta: learning rate (0 < eta <= 1)
    * n_iter: number of iterations
    * random_state: seed value
    
    attributes:
    * w_: 1 dim array, weights
    * cost_: list, sse of cost func of each epoch
    
    """
    
    def __init__(self, eta = 0.01, n_iter = 50, random_state = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        
    def fit(self, X, y):
        """
        fit to training data
        
        parameters:
        * X: independent variables
        * y: target variable
        
        return:
        * self: object
        """
        
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc = 0.0, scale = 0.01, size = X.shape[1] + 1)
        self.cost_ = []
        
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            # calculate an error
            errors = (y - output)
            # update ws
            # ds = eta * sum_i(y^i - phi(z^i))x_j^i (j = 1,...,m)
            self.w_[1:] += self.eta * X.T.dot(errors)
            # update w_0
            # dw_0 = eta * sum_i(y^i - phi(z^i))
            self.w_[0] = self.eta * errors.sum()
            # calculate cost function's value
            cost = (errors**2).sum() / 2.0
            # store cost in self.cost_
            self.cost_.append(cost)
        return self
    
    def net_input(self, X):
        """calculate total inputs"""
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def activation(self, X):
        """ calculate outputs of linear activation function """
        return X
    
    def predict(self, X):
        """ return class labels in the next step"""
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)
            
        
        
class AdalineSGD(object):
    """ADAptive LInear NEuron classifier with Stochastic Gradient Descent method
    
    parameters:
    * eta: learning rate
    * n_iter: number of iteration
    * shuffle: boolean (default: True), in every epoch, shuffle order of training data to avoid circulations of learning
    * random_state: the number of random seed
    
    attributes:
    * w_: 1 dim array, weights
    * cost_: the value of sse calculated in each epoch
    
    """
    
    def __init__(self, eta = 0.01, n_iter = 10, shuffle = True, random_state = None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        self.random_state = random_state
        
    def fit(self, X, y):
        """
        fitting to training data
        
        parameters:
        * X: independent variables
        * y: target variable
        
        return:
        * self: object
        
        """
        
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost)/len(y)
            self.cost_.append(avg_cost)
        return self
    
    def partial_fit(self, X, y):
        """ calculate weights with given training data without initializing weights """
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        # if target y has more than two unique elements, update with each xi and target
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_wegihts(xi, target)
        # if target y has ONE unique element, update with whole xis and targets
        else:
            self._update_weights(X, y)
        return self
    
    def _shuffle(self, X, y):
        """ shuffle training data """
        r = self.rgen.permutation(len(y))
        return X[r], y[r]
    
    def _initialize_weights(self, m):
        """ initialze weights as small random numbers """
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc = .0, scale = .01, size = m + 1)
        self.w_initialized = True
        
    def _update_weights(self, xi, target):
        """ update with learning rules of ADALINE """
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error ** 2 # cost function value that is calculated by previous weights
        return cost
    
    def net_input(self, X):
        """ calc total input """
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def activation(self, X):
        """ calc output of activation function """
        return X
    
    def predict(self, X):
        """" return class label of the next step """
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)