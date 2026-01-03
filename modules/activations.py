from backend import xp
def gelu(x):
    return 0.5 * x * (1 + xp.tanh(xp.sqrt(2 / xp.pi) * (x + 0.044715 * xp.power(x, 3))))

def gelu_derivative(x):
    k = xp.sqrt(2 / xp.pi)
    x3 = 0.044715 * x**3
    x2 = 0.044715 * x**2
    inner = k * (x + x3)
    tanh_inner = xp.tanh(inner)
    
    return 0.5 * (1 + tanh_inner) + 0.5 * x * (1 - tanh_inner**2) * k * (1 + 3 * x2)

def relu(x):
    return xp.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x, axis=-1):
    exp_x = xp.exp(x - xp.max(x, axis=axis, keepdims=True))
    return exp_x / xp.sum(exp_x, axis=axis, keepdims=True)

def softmax_derivative(softmax_output, gradient, axis=-1):
    dx = softmax_output * gradient
    sum_dx = xp.sum(dx, axis=axis, keepdims=True)
    return dx - softmax_output * sum_dx