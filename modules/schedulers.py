import backend as xp

class LRScheduler:
    def __init__(self, optimizer, last_step=-1):
        self.optimizer = optimizer
        for group in optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
        self.base_lrs = [g['initial_lr'] for g in optimizer.param_groups]
        self.last_step = last_step

    def update_groups(self, new_lrs):
        for group, lr in zip(self.optimizer.param_groups, new_lrs):
            group['lr'] = float(lr)
    
    @property
    def lr(self):
        return self.optimizer.param_groups[0]['lr']
        
class StepLR(LRScheduler):
    def __init__(self, optimizer, step_size, gamma=0.1, last_step=-1):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_step)

    def step(self):
        self.last_step += 1
        decay = self.gamma ** (self.last_step // self.step_size)
        self.update_groups([base_lr * decay for base_lr in self.base_lrs])

class CosineAnnealingLR(LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0, last_step=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_step)

    def step(self):
        self.last_step += 1
        t_cur = self.last_step % self.T_max
        cos_out = xp.cos(xp.pi * t_cur / self.T_max)
        
        new_lrs = [
            self.eta_min + 0.5 * (base_lr - self.eta_min) * (1 + cos_out)
            for base_lr in self.base_lrs
        ]
        self.update_groups(new_lrs)

class ReduceLROnPlateau(LRScheduler):
    def __init__(self, optimizer, factor=0.1, patience=20, mode='min', min_lr=1e-6):
        super().__init__(optimizer)
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.mode = mode
        self.best = xp.inf if mode == 'min' else -xp.inf
        self.stagnation = 0

    def step(self, metric):
        self.last_step += 1
        
        is_better = metric < self.best if self.mode == 'min' else metric > self.best
        
        if is_better:
            self.best = metric
            self.stagnation = 0
            return

        self.stagnation += 1
        if self.stagnation >= self.patience:
            new_lrs = [max(g['lr'] * self.factor, self.min_lr) for g in self.optimizer.param_groups]
            self.update_groups(new_lrs)
            self.stagnation = 0