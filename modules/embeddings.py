from backend import xp

class Embedding:
    def __init__(self, vocab_size, embed_dim):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.weights = xp.random.randn(vocab_size, embed_dim) * 0.02
        self.params = [self.weights]
        

    def forward(self, x):
        self.x = x
        return self.weights[x]

    def __call__(self, x):
        return self.forward(x)

    def backward(self, d_out):
        self.grads = xp.zeros_like(self.weights)
        x_flat = self.x.reshape(-1) 

        d_out_flat = d_out.reshape(-1, self.embed_dim)

        xp.add.at(self.grads, x_flat, d_out_flat)
            
        return None

    def get_grads(self):
        return [self.grads]
