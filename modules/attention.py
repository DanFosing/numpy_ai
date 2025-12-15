import numpy as np
from .activations import softmax

class Attention: # attn_type = "gqa" or "mha" or "mqa"
    def __init__(self, embed_dim, query_heads, kv_heads, attn_type="gqa"):
        self.embed_dim = embed_dim
        self.query_heads = query_heads
        self.kv_heads = kv_heads
        assert embed_dim % query_heads == 0, "Embedding dimension must be divisible by the number of query heads"
        assert embed_dim % kv_heads == 0, "Embedding dimension must be divisible by the number of key-value heads"
        assert query_heads % kv_heads == 0, "Query heads must be divisible by the number of key-value heads"
        
        self.head_dim = embed_dim // query_heads
        if attn_type == "gqa":
            self.group_size = query_heads // kv_heads
        elif attn_type == "mha":
            assert query_heads == kv_heads, "Query heads must equal KV heads for MHA"
            self.group_size = 1
        elif attn_type == "mqa":
            assert kv_heads == 1, "For MQA, kv_heads must be 1" 
            self.group_size = query_heads
        else:
            raise ValueError(f"Invalid attention type: {attn_type}")

        
        scale = (1.0 / np.sqrt(embed_dim))
        
        self.Wq_proj = np.random.randn(embed_dim, embed_dim) * scale
        self.Wk_proj = np.random.randn(embed_dim, kv_heads * self.head_dim) * scale
        self.Wv_proj = np.random.randn(embed_dim, kv_heads * self.head_dim) * scale
        self.Wout_proj = np.random.randn(embed_dim, embed_dim) * scale

        self.params = [self.Wq_proj, self.Wk_proj, self.Wv_proj, self.Wout_proj]
    def __call__(self, x):
        return self.forward(x)
    def forward(self, X, mask=None):
        batch_size, seq_length, embed_dim = X.shape
        Q = np.matmul(X, self.Wq_proj)
        K = np.matmul(X, self.Wk_proj)
        V = np.matmul(X, self.Wv_proj)
        
        Q = Q.reshape(batch_size, seq_length, self.query_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_length, self.kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_length, self.kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        if self.group_size > 1:
            K_expanded = np.repeat(K, self.group_size, axis=1)
            V_expanded = np.repeat(V, self.group_size, axis=1)
        else:
            K_expanded = K
            V_expanded = V
        scores = np.matmul(Q, K_expanded.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)

        if mask is not None:
            scores += mask * -1e9 # not using -np.inf to avoid instabilities if all tokens are masked
        attn_weights = softmax(scores, axis=-1)
        
        attn_output = np.matmul(attn_weights, V_expanded)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_length, embed_dim)
        self.cache = {"X": X, "Q": Q, "K": K, "V": V, "K_expanded": K_expanded, "V_expanded": V_expanded, "attn_weights": attn_weights, "attn_output": attn_output}

        output = np.matmul(attn_output, self.Wout_proj)
        return output
        
    def backward(self, gradient):
            X = self.cache["X"]
            Q = self.cache["Q"]
            K_expanded = self.cache["K_expanded"]
            V_expanded = self.cache["V_expanded"]
            attn_weights = self.cache["attn_weights"]
            attn_output = self.cache["attn_output"]

            batch_size, seq_length, embed_dim = X.shape
            
            dWout_proj = np.matmul(attn_output.swapaxes(-1, -2), gradient).sum(axis=0)
            
            dattn_output = np.matmul(gradient, self.Wout_proj.T)
            dattn_output = dattn_output.reshape(batch_size, seq_length, self.query_heads, self.head_dim).transpose(0, 2, 1, 3)

            dV_expanded = np.matmul(attn_weights.transpose(0, 1, 3, 2), dattn_output)
            dA = np.matmul(dattn_output, V_expanded.transpose(0, 1, 3, 2))
            
            dscores = attn_weights * (dA - np.sum(dA * attn_weights, axis=-1, keepdims=True)) / np.sqrt(self.head_dim)
            
            dQ = np.matmul(dscores, K_expanded)
            dK_expanded = np.matmul(dscores.transpose(0, 1, 3, 2), Q)
            
            if self.group_size > 1:
                dK = dK_expanded.reshape(batch_size, self.kv_heads, self.group_size, seq_length, self.head_dim).sum(axis=2)
                dV = dV_expanded.reshape(batch_size, self.kv_heads, self.group_size, seq_length, self.head_dim).sum(axis=2)
            else:
                dK = dK_expanded
                dV = dV_expanded

            dQ = dQ.transpose(0, 2, 1, 3).reshape(batch_size * seq_length, self.query_heads * self.head_dim)
            dK = dK.transpose(0, 2, 1, 3).reshape(batch_size * seq_length, self.kv_heads * self.head_dim)
            dV = dV.transpose(0, 2, 1, 3).reshape(batch_size * seq_length, self.kv_heads * self.head_dim)
            
            X_flat = X.reshape(batch_size * seq_length, embed_dim)
            
            dWq_proj = np.matmul(X_flat.T, dQ)
            dWk_proj = np.matmul(X_flat.T, dK)
            dWv_proj = np.matmul(X_flat.T, dV)
            
            dX = np.matmul(dQ, self.Wq_proj.T)
            dX += np.matmul(dK, self.Wk_proj.T)
            dX += np.matmul(dV, self.Wv_proj.T)
            dX = dX.reshape(batch_size, seq_length, embed_dim)
            
            self.grads = [dWq_proj, dWk_proj, dWv_proj, dWout_proj]
            return dX
