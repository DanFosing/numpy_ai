import backend as xp

class Linear:
    def __init__(self, input_size, output_size, bias=True):
        scale = xp.sqrt(2 / input_size)
        self.weights = xp.random.randn(input_size, output_size) * scale
        self.bias = xp.zeros(output_size) if bias else None
        
        self.params = [self.weights] + ([self.bias] if bias else [])
        self.x = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.x = x
        out = xp.dot(x, self.weights)
        if self.bias is not None:
            out += self.bias
        return out

    def backward(self, gradient):
        if self.x.ndim > 2:
            x_flat = self.x.reshape(-1, self.x.shape[-1])
            grad_flat = gradient.reshape(-1, gradient.shape[-1])
            dw = xp.dot(x_flat.T, grad_flat)
            dx = xp.dot(grad_flat, self.weights.T).reshape(self.x.shape)
        else:
            dw = xp.dot(self.x.T, gradient)
            dx = xp.dot(gradient, self.weights.T)

        if self.bias is not None:
            axes = tuple(range(gradient.ndim - 1))
            db = xp.sum(gradient, axis=axes)
            self.grads = [dw, db]
        else:
            self.grads = [dw]

        return dx

    def get_grads(self):
        return self.grads