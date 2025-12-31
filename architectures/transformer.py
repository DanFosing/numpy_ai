import backend as xp
from modules.attention import Attention
from modules.mlp import MLP
from modules.layer_norm import LayerNorm
from modules.linear import Linear
from modules.embeddings import Embedding
from modules.rope import apply_rotary_emb, precompute_freqs_cis

class TransformerLayer:
    def __init__(self, embed_dim, query_heads, kv_heads, hidden_dim, dropout_rate=0.0):
        self.ln1 = LayerNorm(embed_dim)
        self.attn = Attention(embed_dim, query_heads, kv_heads)
        self.ln2 = LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, hidden_dim, dropout_rate)
        
        self.params = self.ln1.params + self.attn.params + self.ln2.params + self.mlp.params
    
    def forward(self, x, freqs_cis, mask=None):
        self.x = x
        self.x_ln1 = self.ln1(x)
        self.attn_output = self.attn(self.x_ln1, freqs_cis=freqs_cis, mask=mask)
        self.x_residual_1 = x + self.attn_output
        
        self.x_ln2 = self.ln2(self.x_residual_1)
        self.mlp_output = self.mlp(self.x_ln2)
        self.output = self.x_residual_1 + self.mlp_output
        return self.output
    
    def backward(self, gradient, freqs_cis):
        
        d_mlp_output = self.mlp.backward(gradient)
        d_ln2_input = self.ln2.backward(d_mlp_output)
        
        d_residual_1 = gradient + d_ln2_input
        
        d_attn_output = self.attn.backward(d_residual_1, freqs_cis=freqs_cis)
        d_ln1_input = self.ln1.backward(d_attn_output)
        
        d_input = d_residual_1 + d_ln1_input
        return d_input

    def get_grads(self):
        return self.ln1.grads + self.attn.grads + self.ln2.grads + self.mlp.grads

    def train(self):
        self.mlp.train()

    def eval(self):
        self.mlp.eval()

class Transformer:
    def __init__(self, vocab_size, embed_dim, query_heads, kv_heads, hidden_dim=None, layers=4, max_seq_len=512, dropout_rate=0.0):
        self.embed_dim = embed_dim
        self.query_heads = query_heads
        
        if hidden_dim is None:
            hidden_dim = 4 * embed_dim
        
        self.hidden_dim = hidden_dim
        
        self.token_embedding = Embedding(vocab_size, embed_dim)
        head_dim = embed_dim // query_heads

        # Computing frequency embeddings on the fly may be slow for a small model so we precompute them
        self.freqs_cis = precompute_freqs_cis(head_dim, max_seq_len)
        
        self.layers = []
        for _ in range(layers):
            layer = TransformerLayer(embed_dim, query_heads, kv_heads, hidden_dim=hidden_dim, dropout_rate=dropout_rate)
            self.layers.append(layer)
            
        self.layer_norm = LayerNorm(embed_dim)
        self.lm_head = Linear(embed_dim, vocab_size, bias=False)
        
        self.params = self.token_embedding.params
        for layer in self.layers:
            self.params.extend(layer.params)
        self.params += self.layer_norm.params + self.lm_head.params

    def forward(self, tokens, mask=None):
        B, T = tokens.shape
        if mask is None:
            mask = xp.triu(xp.ones((T, T)), k=1).reshape(1, 1, T, T)
            
        x = self.token_embedding(tokens)
        
        for layer in self.layers:
            x = layer.forward(x, self.freqs_cis, mask)
            
        x = self.layer_norm(x)
        logits = self.lm_head(x)
        
        self.output = logits
        return self.output
    
    def backward(self, grad_logits):
        d_x = self.lm_head.backward(grad_logits)
        d_x = self.layer_norm.backward(d_x)
        
        for layer in reversed(self.layers):
            d_x = layer.backward(d_x, self.freqs_cis)
            
        self.token_embedding.backward(d_x)
        
        self.grads = self.token_embedding.get_grads()
        for layer in self.layers:
            self.grads.extend(layer.get_grads())
        self.grads += self.layer_norm.grads + self.lm_head.grads
        
        return None

    def get_grads(self):
        return self.grads
    
    def train(self):
        for layer in self.layers:
            layer.train()

    def eval(self):
        for layer in self.layers:
            layer.eval()
        
    def state_dict(self, complete=False): # Complete state_dict contains all config info, whereas complete=False only contains weights (PyTorch-style)
        state = {
            'weights': {
                'token_embedding': self.token_embedding.params,
                'layers': [
                    {
                        'ln1': layer.ln1.params,
                        'attn': layer.attn.params,
                        'ln2': layer.ln2.params,
                        'mlp': layer.mlp.params
                    }
                    for layer in self.layers
                ],
                'layer_norm': self.layer_norm.params,
                'lm_head': self.lm_head.params
            }
        }
        
        if complete:
            state['config'] = {
                'vocab_size': self.token_embedding.params[0].shape[0],
                'embed_dim': self.embed_dim,
                'query_heads': self.query_heads,
                'kv_heads': self.layers[0].attn.kv_heads,
                'hidden_dim': self.hidden_dim,
                'layers': len(self.layers),
                'max_seq_len': len(self.freqs_cis),
                'dropout_rate': self.layers[0].mlp.dropout_rate
            }
        
        return state

    def load_state_dict(self, state):
        weights = state.get('weights', state)
        
        # Validate structure
        if len(weights['layers']) != len(self.layers):
            raise ValueError(f"State dict has {len(weights['layers'])} layers, model has {len(self.layers)}")
    
        if weights['token_embedding'][0].shape != self.token_embedding.params[0].shape:
            raise ValueError(f"Embedding shape mismatch: state has {weights['token_embedding'][0].shape}, "
                            f"model has {self.token_embedding.params[0].shape}")
        
        if weights['lm_head'][0].shape != self.lm_head.params[0].shape:
            raise ValueError(f"LM head shape mismatch: state has {weights['lm_head'][0].shape}, "
                            f"model has {self.lm_head.params[0].shape}")
        
        # Load weights
        self.token_embedding.params = weights['token_embedding']
        
        for layer, layer_state in zip(self.layers, weights['layers']):
            layer.ln1.params = layer_state['ln1']
            layer.attn.params = layer_state['attn']
            layer.ln2.params = layer_state['ln2']
            layer.mlp.params = layer_state['mlp']
        
        self.layer_norm.params = weights['layer_norm']
        self.lm_head.params = weights['lm_head']
        
        self.params = self.token_embedding.params[:]
        for layer in self.layers:
            self.params.extend(layer.params)
        self.params += self.layer_norm.params + self.lm_head.params

    @classmethod
    def from_state_dict(cls, state): # Initialize model from complete state_dict
        if 'config' not in state:
            raise ValueError("State dict must include 'config' to use from_state_dict()")
        
        config = state['config']
        model = cls(
            vocab_size=config['vocab_size'],
            embed_dim=config['embed_dim'],
            query_heads=config['query_heads'],
            kv_heads=config['kv_heads'],
            hidden_dim=config['hidden_dim'],
            layers=config['layers'],
            max_seq_len=config['max_seq_len'],
            dropout_rate=config['dropout_rate']
        )
        model.load_state_dict(state)
        return model