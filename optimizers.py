import numpy as np

class Optimizer:
    def step(self, params, grads):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, params, grads):
        return params - self.lr * grads


class Momentum(Optimizer):
    def __init__(self, lr=0.01, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.v = np.zeros(2)

    def step(self, params, grads):
        self.v = self.beta * self.v + grads
        return params - self.lr * self.v


class Nesterov(Optimizer):
    def __init__(self, lr=0.01, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.v = np.zeros(2)

    def step(self, params, grads):
        self.v = self.beta * self.v + grads
        return params - self.lr * (self.beta * self.v + grads)


class AdaGrad(Optimizer):
    def __init__(self, lr=0.1, eps=1e-8):
        self.lr = lr
        self.eps = eps
        self.h = np.zeros(2)

    def step(self, params, grads):
        self.h += grads ** 2
        return params - self.lr * grads / (np.sqrt(self.h) + self.eps)


class RMSProp(Optimizer):
    def __init__(self, lr=0.01, beta=0.9, eps=1e-8):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.h = np.zeros(2)

    def step(self, params, grads):
        self.h = self.beta * self.h + (1 - self.beta) * grads ** 2
        return params - self.lr * grads / (np.sqrt(self.h) + self.eps)


class Adam(Optimizer):
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = np.zeros(2)
        self.v = np.zeros(2)
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)

        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)