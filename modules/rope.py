from backend import xp

def precompute_freqs_cis(dim, end, theta=10000.0):
    inv_freq = 1.0 / (theta ** (xp.arange(0, dim, 2) / dim))
    
    freqs = xp.arange(end)[:, None] * inv_freq
    
    return xp.exp(1j * freqs)

def apply_rotary_emb(x, freqs_cis):
    freqs = freqs_cis[None, None, :x.shape[2], :]

    z = x[..., 0::2] + 1j * x[..., 1::2]

    z_rotated = z * freqs

    out = xp.empty_like(x)
    out[..., 0::2] = z_rotated.real
    out[..., 1::2] = z_rotated.imag
    
    return out