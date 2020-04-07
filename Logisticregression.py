import numpy as np

class LogisticRegressionGD(object):
    """ Logistic regression classifier with Gradient descent method
    
    parameters:
    * eta: learning rate
    * n_iter: number of epoch
    * random_state: the number of random seed
    
    attributes:
    * w_: 1 dim array, weights(coefficients of regression)
    * cost_: list, sse of each step
    
    """
    
    def __init__(self, eta = 0.05, n_iter = 100, random_state = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        
    def fit(self, X, y):
        """ learning with training data
        
        parameters:
        * X: independent variables
        * y: target variable
        
        return:
        * self: object
        
        """
        
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc = 0.0, scale = 0.01, size = 1 + X.shape[1])
        self.cost_ = []
        
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            
            cost = -y.dot(np.log(output)) - (1 - y).dot(np.log(1 - output))
            self.cost_.append(cost)
        return self
    
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def activation(self, z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))
    
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)
        