import backend as xp

class DataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(dataset)
        self.drop_last = drop_last

    def __iter__(self):
        indices = xp.arange(self.num_samples)
        if self.shuffle:
            indices = xp.random.permutation(self.num_samples)
        
        num_batches = self.num_samples // self.batch_size
        if not self.drop_last and self.num_samples % self.batch_size != 0:
            num_batches += 1
        
        for i in range(num_batches):
            start = i * self.batch_size
            end = min(start + self.batch_size, self.num_samples)
            
            batch_indices = indices[start:end]
            batch = [self.dataset[int(idx)] for idx in batch_indices]
            
            yield tuple(xp.stack([item[j] for item in batch]) 
                       for j in range(len(batch[0])))

    def __len__(self):
        if self.drop_last:
            return self.num_samples // self.batch_size
        return (self.num_samples + self.batch_size - 1) // self.batch_size