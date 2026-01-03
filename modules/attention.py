from backend import xp
from .activations import softmax
from .rope import apply_rotary_emb

class Attention:
    def __init__(self, embed_dim, query_heads, kv_heads):
        self.embed_dim = embed_dim
        self.query_heads = query_heads
        self.kv_heads = kv_heads
        assert embed_dim % query_heads == 0, "Embedding dimension must be divisible by the number of query heads"
        assert embed_dim % kv_heads == 0, "Embedding dimension must be divisible by the number of key-value heads"
        assert query_heads % kv_heads == 0, "Query heads must be divisible by the number of key-value heads"
        
        self.head_dim = embed_dim // query_heads
        # Attention types:
        # - MHA: query_heads == kv_heads
        # - MQA: kv_heads == 1 and query_heads > 1
        # - GQA: query_heads > kv_heads and kv_heads > 1
        self.group_size = query_heads // kv_heads

        
        scale = (1.0 / xp.sqrt(self.head_dim))
        
        # Manual Q, K, V, and output projections (instead of linear layers which would've been more compact
        # and simpler to implement) are used to clearly show how attention works internally
        self.Wq_proj = xp.random.randn(embed_dim, embed_dim) * scale
        self.Wk_proj = xp.random.randn(embed_dim, kv_heads * self.head_dim) * scale
        self.Wv_proj = xp.random.randn(embed_dim, kv_heads * self.head_dim) * scale
        self.Wout_proj = xp.random.randn(embed_dim, embed_dim) * scale

        self.params = [self.Wq_proj, self.Wk_proj, self.Wv_proj, self.Wout_proj]

    def __call__(self, x, freqs_cis=None, mask=None):
        return self.forward(x, freqs_cis, mask)

    def forward(self, X, freqs_cis=None, mask=None):
        batch_size, seq_length, embed_dim = X.shape
        Q = xp.matmul(X, self.Wq_proj)
        K = xp.matmul(X, self.Wk_proj)
        V = xp.matmul(X, self.Wv_proj)
        
        Q = Q.reshape(batch_size, seq_length, self.query_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_length, self.kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_length, self.kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        if freqs_cis is not None:
            Q = apply_rotary_emb(Q, freqs_cis)
            K = apply_rotary_emb(K, freqs_cis)

        if self.group_size > 1:
            K_expanded = xp.repeat(K, self.group_size, axis=1)
            V_expanded = xp.repeat(V, self.group_size, axis=1)
        else:
            K_expanded = K
            V_expanded = V
        scores = xp.matmul(Q, K_expanded.transpose(0, 1, 3, 2)) / xp.sqrt(self.head_dim)

        if mask is not None:
            scores += mask * -1e9 # not using -xp.inf to avoid instabilities if all tokens are masked
        attn_weights = softmax(scores, axis=-1)
        
        attn_output = xp.matmul(attn_weights, V_expanded)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_length, embed_dim)
        self.cache = {"X": X, "Q": Q, "K": K, "V": V, "K_expanded": K_expanded, "V_expanded": V_expanded, "attn_weights": attn_weights, "attn_output": attn_output}

        output = xp.matmul(attn_output, self.Wout_proj)
        return output
        
    def backward(self, gradient, freqs_cis=None):
            X = self.cache["X"]
            Q = self.cache["Q"]
            K_expanded = self.cache["K_expanded"]
            V_expanded = self.cache["V_expanded"]
            attn_weights = self.cache["attn_weights"]
            attn_output = self.cache["attn_output"]

            batch_size, seq_length, embed_dim = X.shape
            
            dWout_proj = xp.matmul(attn_output.swapaxes(-1, -2), gradient).sum(axis=0)
            
            dattn_output = xp.matmul(gradient, self.Wout_proj.T)
            dattn_output = dattn_output.reshape(batch_size, seq_length, self.query_heads, self.head_dim).transpose(0, 2, 1, 3)

            dV_expanded = xp.matmul(attn_weights.transpose(0, 1, 3, 2), dattn_output)
            dA = xp.matmul(dattn_output, V_expanded.transpose(0, 1, 3, 2))
            
            dscores = attn_weights * (dA - xp.sum(dA * attn_weights, axis=-1, keepdims=True)) / xp.sqrt(self.head_dim)
            
            dQ = xp.matmul(dscores, K_expanded)
            dK_expanded = xp.matmul(dscores.transpose(0, 1, 3, 2), Q)
            
            if freqs_cis is not None:
                dQ = apply_rotary_emb(dQ, xp.conj(freqs_cis))
                dK_expanded = apply_rotary_emb(dK_expanded, xp.conj(freqs_cis))
            
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
            
            dWq_proj = xp.matmul(X_flat.T, dQ)
            dWk_proj = xp.matmul(X_flat.T, dK)
            dWv_proj = xp.matmul(X_flat.T, dV)
            
            dX = xp.matmul(dQ, self.Wq_proj.T)
            dX += xp.matmul(dK, self.Wk_proj.T)
            dX += xp.matmul(dV, self.Wv_proj.T)
            dX = dX.reshape(batch_size, seq_length, embed_dim)
            
            self.grads = [dWq_proj, dWk_proj, dWv_proj, dWout_proj]
            return dX
