import backend as xp
from backend import USE_GPU, get_device, to_cpu, to_gpu
import time
from sklearn import datasets
from sklearn.model_selection import train_test_split
from modules.activations import relu, relu_derivative, softmax

def cross_entropy_loss(y, y_pred, eps=1e-9):
    return -xp.sum(y * xp.log(y_pred + eps)) / y.shape[0]

def one_hot_encode(y, num_classes=10): 
    one_hot = xp.zeros((y.shape[0], num_classes))
    one_hot[xp.arange(y.shape[0]), y] = 1
    return one_hot 

mnist = datasets.load_digits()
X = mnist.data
y = mnist.target
y_encoded = one_hot_encode(y) #one hot encoding for simplification and better accuracy

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Sklearn always returns numpy arrays, if we want to use GPU we need to convert them
if get_device(X_train) == "cpu" and USE_GPU:
    X_train = to_gpu(X_train)
    X_test = to_gpu(X_test)
    y_train = to_gpu(y_train)
    y_test = to_gpu(y_test)

class MLP:
    def __init__(self, input_size, hidden_size, output_size, lr=0.01):
        self.weights_input_hidden = xp.random.randn(input_size, hidden_size) * xp.sqrt(2 / input_size) # He initialization
        self.weights_hidden_output = xp.random.randn(hidden_size, output_size) * xp.sqrt(2 / hidden_size) # He initialization
        self.bias_hidden = xp.zeros((1, hidden_size))
        self.bias_output = xp.zeros((1, output_size))
        self.lr = lr

    def forward(self, X):
        self.hidden_input = xp.matmul(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden = relu(self.hidden_input)
        self.output_input = xp.matmul(self.hidden, self.weights_hidden_output) + self.bias_output
        self.output = softmax(self.output_input, axis=1)
        return self.output
    
    def backward(self, X, y):
        batch_size = y.shape[0]
        # the combination of softmax and cross-entropy loss simplifies the gradient to this:
        output_delta = self.output - y
        hidden_delta = xp.matmul(output_delta, self.weights_hidden_output.T) * relu_derivative(self.hidden_input)
        
        self.weights_hidden_output -= xp.matmul(self.hidden.T, output_delta) / batch_size * self.lr
        self.weights_input_hidden -= xp.matmul(X.T, hidden_delta) / batch_size * self.lr

        self.bias_output -= xp.sum(output_delta, axis=0, keepdims=True) / batch_size * self.lr
        self.bias_hidden -= xp.sum(hidden_delta, axis=0, keepdims=True) / batch_size * self.lr
    
    def train(self, X, y, epochs=10000):
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y)
            if epoch % 200 == 0:
                loss = cross_entropy_loss(y, self.output)
                print(f'Epoch {epoch}, Loss: {loss}')

start_time = time.time()
mlp = MLP(input_size=64, hidden_size=128, output_size=10, lr=0.1)
mlp.train(X_train, y_train, epochs=5000)
print(f'Time taken: {time.time() - start_time:.2f} seconds')
# Test
y_pred_raw = mlp.forward(X_test)
y_pred = xp.argmax(y_pred_raw, axis=1)
accuracy = xp.mean(y_pred == xp.argmax(y_test, axis=1))
print(f'Test Accuracy: {accuracy * 100:.2f}%')