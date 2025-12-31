import backend as xp

class AdamW:
    def __init__(self, params, lr=3e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, decay_on_1d=False):
        if isinstance(params, list) and len(params) > 0 and not isinstance(params[0], dict):
            params = [{'params': params}]

        self.decay_on_1d = decay_on_1d 
        self.param_groups = []
        for group in params:
            self.param_groups.append({
                'params': group['params'],
                'lr': group.get('lr', lr),
                'betas': group.get('betas', betas),
                'eps': group.get('eps', eps),
                'weight_decay': group.get('weight_decay', weight_decay),
            })

        self.state = {}
        for group in self.param_groups:
            for p in group['params']:
                self.state[id(p)] = {
                    'step': 0,
                    'm': xp.zeros_like(p),
                    'v': xp.zeros_like(p),
                }

    def step(self, grads_list=None):
        k = 0
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            
            for p in group['params']:
                # Use p.grad if available, otherwise use grads_list
                if hasattr(p, 'grad') and p.grad is not None:
                    grad = p.grad
                elif grads_list is not None:
                    grad = grads_list[k]
                    k += 1
                else:
                    raise ValueError("No gradient found for parameter")

                state = self.state[id(p)]
                state['step'] += 1
                t = state['step']
                m, v = state['m'], state['v']
                
                # Weight Decay
                if wd > 0 and (p.ndim > 1 or self.decay_on_1d):
                    p -= lr * wd * p
                
                # Update momentums
                m[:] = beta1 * m + (1 - beta1) * grad
                v[:] = beta2 * v + (1 - beta2) * grad**2
                
                # Bias correction
                m_hat = m / (1 - beta1**t)
                v_hat = v / (1 - beta2**t)
                
                # Update parameters
                p -= lr * m_hat / (xp.sqrt(v_hat) + eps)
    
    def zero_grad(self, grads_list=None):
        if grads_list:
            for g in grads_list:
                if g is not None: 
                    g.fill(0.0)
            return

        for group in self.param_groups:
            for p in group['params']:
                if hasattr(p, 'grad') and p.grad is not None:
                    p.grad.fill(0.0)