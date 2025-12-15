import numpy as np
from .activations import gelu, gelu_derivative

class MLP:
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.0):

        scale = np.sqrt(2.0 / input_dim)
        self.W_fc = np.random.randn(input_dim, hidden_dim) * scale
        self.b_fc = np.zeros(hidden_dim)

        self.W_proj = np.random.randn(hidden_dim, input_dim) * scale
        self.b_proj = np.zeros(input_dim)
        
        self.dropout_rate = dropout_rate
        self.params = [self.W_fc, self.b_fc, self.W_proj, self.b_proj]
    def __call__(self, x):
        return self.forward(x)
    def forward(self, x):
        self.input = x
        
        self.hidden = np.matmul(x, self.W_fc) + self.b_fc
        
        self.activated = gelu(self.hidden)
        
        self.output = np.matmul(self.activated, self.W_proj) + self.b_proj
        
        if self.dropout_rate > 0:
            self.dropout_mask = (np.random.rand(*self.output.shape) > self.dropout_rate).astype(float)
            scale = 1.0 / (1.0 - self.dropout_rate)
            self.output = self.output * self.dropout_mask * scale
        
        return self.output

    def backward(self, gradient):
        axes = tuple(range(gradient.ndim - 1))

        if self.dropout_rate > 0:
            scale = 1.0 / (1.0 - self.dropout_rate)
            gradient = gradient * self.dropout_mask * scale

        dW_proj = np.matmul(self.activated.swapaxes(-1, -2), gradient).sum(axis=0)
        db_proj = np.sum(gradient, axis=axes)
        dactivated = np.matmul(gradient, self.W_proj.T)
        
        dhidden = dactivated * gelu_derivative(self.hidden)
        
        dW_fc = np.matmul(self.input.swapaxes(-1, -2), dhidden).sum(axis=0)
        db_fc = np.sum(dhidden, axis=axes)
        dinput = np.matmul(dhidden, self.W_fc.T)
        
        self.grads = [dW_fc, db_fc, dW_proj, db_proj]
        
        return dinput