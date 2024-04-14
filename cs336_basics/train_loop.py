import numpy as np
import torch

def get_batch(dataset, batch_size, context_length, device):
    data_batch = []
    label_batch = []
    for i in range(batch_size):
        start_index = np.random.randint(1, len(dataset) - context_length + 1)
        sample = dataset[start_index : start_index + context_length]
        label_batch += (sample,)
        sample2 = sample.copy()-1
        data_batch += (sample2,)
    data_batch = torch.tensor(np.array(data_batch), dtype=torch.long, device=device)
    label_batch = torch.tensor(np.array(label_batch), dtype=torch.long, device=device)
    return (data_batch, label_batch)