from base import *
from dataloader import DataLoader

class Module:
    def __init__(self):
        self.parameters = []
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError("Module forward() method must be implemented by subclasses.")
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def loss(self, *args, **kwargs):
        raise NotImplementedError("Module loss() method must be implemented by subclasses.")
    
class LinearLayer(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W = Tensor(np.random.randn(in_features, out_features) * np.sqrt(2. / in_features), requires_grad=True)
        self.b = Tensor(np.zeros(out_features), requires_grad=True)
        self.parameters = [self.W, self.b]
    
    def forward(self, x):
        return x @ self.W + self.b
    
    
def MSELoss(pred, target):
    return sum((pred - target)* (pred - target)) / pred.data.shape[0]

def CELoss(logits, target):
    log_probs = log_softmax(logits, axis=1)      # 返回 Tensor
    loss = -sum(target * log_probs) / logits.data.shape[0]
    return loss