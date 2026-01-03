from backend import xp

def clip_grad_norm(grads, max_norm):
    sq_sums = [xp.sum(g**2) for g in grads if g is not None]
    if not sq_sums:
        return 0.0
    
    total_norm = xp.sqrt(xp.sum(xp.array(sq_sums)))
    
    if total_norm > max_norm:
        coeff = max_norm / total_norm
        for g in grads:
            if g is not None:
                g *= coeff
                
    return float(total_norm) 
