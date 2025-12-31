import backend as xp

class LayerNorm:
    def __init__(self, dim, eps=1e-8):
        self.gamma = xp.ones(dim)
        self.beta = xp.zeros(dim)
        self.eps = eps

        self.params = [self.gamma, self.beta]
        
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.x = x
        
        self.mu = xp.mean(self.x, axis=-1, keepdims=True)
        self.var = xp.var(self.x, axis=-1, keepdims=True)
        self.sigma = xp.sqrt(self.var + self.eps)
        
        self.x_normalized = (self.x - self.mu) / self.sigma
        
        out = self.gamma * self.x_normalized + self.beta
        return out

    def backward(self, gradient):
        D = gradient.shape[-1]
        
        axes = tuple(range(gradient.ndim - 1))

        d_gamma = xp.sum(gradient * self.x_normalized, axis=axes)
        d_beta = xp.sum(gradient, axis=axes)

        dx_normalized = gradient * self.gamma

        mean_correction = xp.sum(dx_normalized, axis=-1, keepdims=True)
        std_correction  = xp.sum(dx_normalized * self.x_normalized, axis=-1, keepdims=True) * self.x_normalized

        dx = (D * dx_normalized - mean_correction - std_correction) / (D * self.sigma)

        self.grads = [d_gamma, d_beta]
        return dx