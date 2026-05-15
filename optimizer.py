from base import *

class Optimizer:
    def __init__(self,param,hyperparams:dict):
        self.param = param
        self.hyperparams = hyperparams
    
    def zero_grad(self):
        for param in self.param:
            param.grad = np.zeros_like(param.grad)
    
    def step(self):
        raise NotImplementedError("Optimizer step() method must be implemented by subclasses.")
    
class SGD(Optimizer):
    '''Stochastic Gradient Descent optimizer with a momentum term.'''
    def __init__(self, param, lr=0.01, beta=0.9):
        super().__init__(param, {'lr': lr, 'beta': beta})
        self.lr = self.hyperparams['lr']
        self.beta = self.hyperparams['beta']
        self.velocity = [np.zeros_like(p.data) for p in self.param]


    def step(self):
        for i, param in enumerate(self.param):
            if param.grad is None:
                continue
            self.velocity[i] = self.beta * self.velocity[i] + (1 - self.beta) * param.grad
            # Update param
            param.data -= self.lr * self.velocity[i]

class Adam(Optimizer):
    '''Adam optimizer.'''
    def __init__(self, param, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(param, {'lr': lr, 'beta1': beta1, 'beta2': beta2, 'eps': eps})
        self.lr = self.hyperparams['lr']
        self.beta1 = self.hyperparams['beta1']
        self.beta2 = self.hyperparams['beta2']
        self.eps = self.hyperparams['eps']
        self.m = [np.zeros_like(p.data) for p in self.param]  # First moment
        self.v = [np.zeros_like(p.data) for p in self.param]  # Second moment
        self.t = 0  # Time step

    def step(self):
        self.t += 1
        for i, param in enumerate(self.param):
            if param.grad is None:
                continue
            grad = param.grad
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
    
    