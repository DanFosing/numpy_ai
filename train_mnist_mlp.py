import urllib.request
import gzip
import struct
import io
import time
from backend import xp
from modules.activations import relu, relu_derivative, softmax

def cross_entropy_loss(y, y_pred, eps=1e-9):
    return -xp.sum(y * xp.log(y_pred + eps)) / y.shape[0]

def one_hot_encode(y, num_classes=10): 
    one_hot = xp.zeros((y.shape[0], num_classes))
    one_hot[xp.arange(y.shape[0]), y] = 1
    return one_hot 

def get_batches(X, y, batch_size=128):
    n = X.shape[0]
    indices = xp.arange(n)
    xp.random.shuffle(indices)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_idx = indices[start:end]
        yield X[batch_idx], y[batch_idx]

urls = {
    'train_images': ["https://systemds.apache.org/assets/datasets/mnist/train-images-idx3-ubyte.gz",
                    "https://azureopendatastorage.blob.core.windows.net/mnist/train-images-idx3-ubyte.gz"],
    'train_labels': ["https://systemds.apache.org/assets/datasets/mnist/train-labels-idx1-ubyte.gz",
                    "https://azureopendatastorage.blob.core.windows.net/mnist/train-labels-idx1-ubyte.gz"],
    'test_images':  ["https://systemds.apache.org/assets/datasets/mnist/t10k-images-idx3-ubyte.gz",
                    "https://azureopendatastorage.blob.core.windows.net/mnist/t10k-images-idx3-ubyte.gz"],
    'test_labels':  ["https://systemds.apache.org/assets/datasets/mnist/t10k-labels-idx1-ubyte.gz",
                    "https://azureopendatastorage.blob.core.windows.net/mnist/t10k-labels-idx1-ubyte.gz"]
}

def download_and_parse(url_list, is_labels=False):
    last_exception = None
    for url in url_list:
        try:
            data = urllib.request.urlopen(url).read()
            f = io.BytesIO(gzip.decompress(data))
            if is_labels:
                struct.unpack(">II", f.read(8))  # skip header
                return xp.frombuffer(f.read(), dtype=xp.uint8)
            else:
                struct.unpack(">IIII", f.read(16))  # skip header
                return xp.frombuffer(f.read(), dtype=xp.uint8).reshape(-1, 784).astype(xp.float32) / 255.0 # normalize
        except Exception as e:
            last_exception = e
            continue
    raise RuntimeError(f"Download failed. Last error: {last_exception}")

X_train = xp.array(download_and_parse(urls['train_images']))
y_train = one_hot_encode(download_and_parse(urls['train_labels'], is_labels=True))
X_test  = xp.array(download_and_parse(urls['test_images']))
y_test  = one_hot_encode(download_and_parse(urls['test_labels'], is_labels=True))


class MLP:
    def __init__(self, input_size, hidden_size, output_size, lr=0.1):
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

    def train(self, X, y, epochs=10, batch_size=128):
        for epoch in range(epochs):
            for X_batch, y_batch in get_batches(X, y, batch_size):
                self.forward(X_batch)
                self.backward(X_batch, y_batch)
            
            full_output = self.forward(X)
            loss = cross_entropy_loss(y, full_output)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

start_time = time.time()
mlp = MLP(input_size=784, hidden_size=128, output_size=10, lr=0.1)
mlp.train(X_train, y_train, epochs=30, batch_size=128)
print(f"Time taken: {time.time() - start_time:.2f} seconds")

y_pred_raw = mlp.forward(X_test)
y_pred = xp.argmax(y_pred_raw, axis=1)
accuracy = xp.mean(y_pred == xp.argmax(y_test, axis=1))
print(f"Test Accuracy: {accuracy * 100:.2f}%")