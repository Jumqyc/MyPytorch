from __future__ import annotations

import numpy as np
from typing import Any, Callable
import matplotlib.pyplot as plt

from collections import deque


def reduce_to_shape(raw_grad, target_shape):
    raw = np.asarray(raw_grad)
    target = tuple(target_shape)
    while raw.ndim > len(target):
        raw = raw.sum(axis=0)
    for axis, (r_dim, t_dim) in enumerate(zip(raw.shape, target)):
        if r_dim != t_dim:
            if t_dim == 1:
                raw = raw.sum(axis=axis, keepdims=True)
            else:
                raise ValueError(f"Cannot reduce axis {axis} with {r_dim} to {t_dim}")
    return raw.reshape(target)



class Tensor:
    def __init__(self,data,
                 requires_grad:bool=False,
                 father_objs:tuple[Tensor, ...] = (), 
                 father_op:Function | None = None,
                 dtype=np.float64):
        self.data = np.array(data, dtype=dtype)
        self.requires_grad = requires_grad
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)
        else:
            self.grad = None

        # for autograd
        self.father_objs = father_objs
        self.father_op = father_op
        self.pending = 0
        self.op_kwargs = {}

    def __getitem__(self, key):
        return self.data[key]
    
    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad}, grad={self.grad})"
    
    def __add__(self, other):
        return add(self, other)
    def __mul__(self, other):
        return mul(self, other)
    def __sub__(self, other):
        return sub(self, other)
    def __neg__(self):
        return self * (-1)
    
    def __truediv__(self, other):
        return truediv(self, other)
    

    def __radd__(self, other):
        return self + other
    def __rmul__(self, other):
        return self * other
    def __rsub__(self, other):
        return  other - self
    def __rtruediv__(self, other):
        return truediv(other, self)
    def __matmul__(self, other):
        return matmul(self, other)
    def __rmatmul__(self, other):
        return matmul(other, self)
        
    def back(self):
        if self.pending != 0:
            raise RuntimeError("back() can only be called on the final output (pending=0).")
        if not self.requires_grad:
            return

        queue = deque([self])
        while queue:
            node = queue.popleft()
            if node.father_op is None:
                continue

            op = node.father_op
            inputs = node.father_objs
            inputs_data = [inp.data for inp in inputs]
            kwargs = node.op_kwargs if hasattr(node, 'op_kwargs') else {}

            for i, father in enumerate(inputs):
                if not father.requires_grad:
                    continue

                if op.vjp is not None:
                    grad_contrib = op.vjp(i, node.grad, *inputs_data, **kwargs)
                elif op.jacobian is not None:
                    local_jac = op.jacobian[i](*inputs_data, **kwargs)
                    grad_contrib = node.grad * local_jac
                else:
                    raise RuntimeError("Operation has neither vjp nor jacobian defined.")
                aligned = reduce_to_shape(grad_contrib, father.data.shape)
                father.grad += aligned
                father.pending -= 1
                if father.pending == 0:
                    queue.append(father)

    def autograd(self):
        if self.pending != 0:
            raise RuntimeError("back() can only be called on a tensor that is the output of the computation graph (pending=0).")
        
        self.grad = np.ones_like(self.data)

        self.back()

    def zerograd(self):
        self.grad = np.zeros_like(self.grad)
        [father.zerograd() for father in self.father_objs if father.requires_grad]
class Function:
    def __init__(self, func, jacobian=None, vjp=None, default_kwargs=None):
        """
        func: forward, f(*inputs_data, **kwargs) -> np.ndarray
        jacobian: tuple of derivative functions for each input, for elementwise operations.
                  jacobian[i](*inputs_data, **kwargs) -> np.ndarray, 
        vjp: vector-Jacobian product function, for more complex operations.
        default_kwargs: default keyword arguments to be used in forward and backward if not provided at call time.
        """
        self.func = func
        self.jacobian = jacobian
        self.vjp = vjp
        self.default_kwargs = default_kwargs if default_kwargs else {}

    def __call__(self, *args, **kwargs):
        merged_kwargs = {**self.default_kwargs, **kwargs}
        tensors = []
        for t in args:
            if not isinstance(t, Tensor):
                t = Tensor(t, requires_grad=False)
            tensors.append(t)
        for t in tensors:
            if t.requires_grad:
                t.pending += 1
        inputs_data = [t.data for t in tensors]
        out_data = self.func(*inputs_data, **merged_kwargs)
        requires_grad = any(t.requires_grad for t in tensors)
        out = Tensor(out_data, requires_grad=requires_grad,
                     father_objs=tuple(tensors), father_op=self)
        out.op_kwargs = merged_kwargs   # 新增属性
        return out

    
add = Function(lambda x,y: x+y, (lambda x,y: 1, lambda x,y: 1))
mul = Function(lambda x,y: x*y, (lambda x,y: y, lambda x,y: x))
sub = Function(lambda x,y: x-y, (lambda x,y: 1, lambda x,y: -1))
truediv = Function(lambda x,y: x/y, (lambda x,y: 1/y, lambda x,y: -x/(y*y)))

sum = Function(lambda x: np.sum(x), (lambda x: np.ones_like(x),))
exp = Function(lambda x: np.exp(x), (lambda x: np.exp(x),))
log = Function(lambda x: np.log(x), (lambda x: 1/(x+1e-8),))

relu = Function(lambda x: np.maximum(0, x), (lambda x: (x > 0).astype(np.float64),))
tanh = Function(lambda x: np.tanh(x), (lambda x: 1 - np.tanh(x)**2,))
sigmoid = Function(lambda x: 1/(1+np.exp(-x)), (lambda x: sigmoid.func(x) * (1 - sigmoid.func(x)),))

# ------------------- matmul -------------------
def matmul_forward(x, y):
    return x @ y

def matmul_vjp(i, grad_output, x, y):
    if i == 0:
        return grad_output @ y.T
    else:
        return x.T @ grad_output

matmul = Function(matmul_forward, vjp=matmul_vjp)

def softmax_func(x, axis=-1):
    shift = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - shift)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def softmax_vjp(i, grad_output, x, axis=-1):
    y = softmax_func(x, axis=axis)
    dot = np.sum(grad_output * y, axis=axis, keepdims=True)
    dx = y * (grad_output - dot)
    return dx

def log_softmax_forward(x, axis=-1):
    shift = np.max(x, axis=axis, keepdims=True)
    return (x - shift) - np.log(np.sum(np.exp(x - shift), axis=axis, keepdims=True))

def log_softmax_vjp(i, grad_output, x, axis=-1):
    shift = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - shift)
    probs = exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    dx = grad_output - probs * np.sum(grad_output, axis=axis, keepdims=True)
    return dx

log_softmax = Function(
    func=log_softmax_forward,
    vjp=log_softmax_vjp,
    default_kwargs={'axis': -1}
)
