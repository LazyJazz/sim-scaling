import torch
import numpy as np
import zarr

class ArrayDataset(torch.utils.data.Dataset):
    def __init__(self, size=64):
        self.array = np.arange(size)

    def __len__(self):
        return self.array.shape[0]

    def __getitem__(self, idx):
        return self.array[idx]
    
if __name__ == "__main__":

    # make dir tmp/
    import os
    if not os.path.exists("tmp"):
        os.makedirs("tmp")
    dataset = ArrayDataset()
    torch_generator = torch.Generator().manual_seed(0)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, generator=torch_generator)
    for i in range(3):
        print(f"--- Epoch {i} ---")
        for batch in dataloader:
            print(batch, dataloader.generator.get_state())
    # save dataloader state
    dataloader_state = dataloader.generator.get_state()
    torch.save(dataloader_state, f"tmp/dataloader_state_epoch_{i}.pt")
    print(f"Saved dataloader state to tmp/dataloader_state_epoch_{i}.pt")

    for i in range(3):
        print(f"--- Epoch {i} ---")
        for batch_idx in range(len(dataloader)):
            print(dataloader[batch_idx])


