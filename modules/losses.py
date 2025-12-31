import backend as xp
from .activations import softmax

class LossModule:
    def __init__(self):
        self.input = None
        self.target = None

    def __call__(self, input, target):
        return self.forward(input, target)

    def forward(self, input, target):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError


class MSELoss(LossModule):
    def forward(self, input, target):
        self.input = input
        self.target = target
        
        return xp.mean((input - target) ** 2)

    def backward(self):
        n = self.input.size
        grad = 2 * (self.input - self.target) / n
        return grad


class MAELoss(LossModule):
    def forward(self, input, target):
        self.input = input
        self.target = target
        return xp.mean(xp.abs(input - target))

    def backward(self):
        n = self.input.size
        grad = xp.sign(self.input - self.target) / n
        return grad


class CrossEntropyLoss(LossModule):
    def __init__(self, eps=1e-9, apply_softmax=True):
        super().__init__()
        self.eps = eps
        self.apply_softmax = apply_softmax
        self.softmax_output = None
        self.original_shape = None

    def forward(self, input, target):
        self.original_shape = input.shape

        if input.ndim > 2: # Flatten for batched inputs
            input = input.reshape(-1, input.shape[-1])
            target = target.reshape(-1)

        self.input = input
        self.target = target


        if self.apply_softmax:
            self.softmax_output = softmax(input, axis=1)
            probs = self.softmax_output
        else:
            probs = input
        
        batch_indices = xp.arange(input.shape[0])
        correct_probs = probs[batch_indices, target]
        
        return -xp.log(correct_probs + self.eps).mean()

    def backward(self):
        B = self.input.shape[0]
        batch_indices = xp.arange(B)
        
        if self.apply_softmax:
            grad = self.softmax_output.copy()
            grad[batch_indices, self.target] -= 1.0 # probability - 1 (1 = 100% would be a perfect prediction)
        else:
            grad = xp.zeros_like(self.input)
            target_probs = self.input[batch_indices, self.target]
            grad[batch_indices, self.target] = -1.0 / (target_probs + self.eps) 
        
        if self.original_shape is not None and len(self.original_shape) > 2:
            grad = grad.reshape(self.original_shape)
        
        return grad / B