from base import *
from typing import Iterable

class DataLoader:
    def __init__(self, X: Tensor, y: Tensor, batch_size: int = 32, shuffle: bool = True):
        assert X.data.shape[0] == len(y.data), "Features and labels must have the same number of samples."
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(self.X.data.shape[0])
        self.current = 0
    
    @property
    def datalength(self):
        return self.X.data.shape[0]

    def __len__(self):
        return (self.datalength + self.batch_size - 1) // self.batch_size
    
    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.current = 0
        return self
    
    def __next__(self):
        if self.current >= self.datalength:
            raise StopIteration
        batch_indices = self.indices[self.current:self.current + self.batch_size]
        batch_X = Tensor(self.X.data[batch_indices], requires_grad=self.X.requires_grad)
        batch_y = Tensor(self.y.data[batch_indices], requires_grad=self.y.requires_grad)
        self.current += self.batch_size
        return batch_X, batch_y
    
    
