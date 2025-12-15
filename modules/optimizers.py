import numpy as np

class AdamW:
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01):
        self.params = params
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        # Initialize momentums
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        # Initialize step counter
        self.t = 0
    
    def step(self, grads):
        self.t += 1
        lr_t = self.lr * np.sqrt(1 - self.betas[1]**self.t) / (1 - self.betas[0]**self.t)
        for i, (param, grad) in enumerate(zip(self.params, grads)):
            if grad is None:
                continue
            # Decoupled weight decay
            if self.weight_decay > 0:
                param -= self.lr * self.weight_decay * param
            # Update momentums
            self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * grad
            self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * grad**2
            # Update parameters
            param -= lr_t * (self.m[i] / (np.sqrt(self.v[i]) + self.eps))