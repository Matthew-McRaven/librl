import itertools

import numpy as np
import torch
import torch.utils.data

# Used to split a single dataset over multple data loaders.
# It's the responsibility of the *caller* to ensure that each item appears only once in the dataset,
# otherwise elements may be in multiple splits.
def load_split_data(dset:torch.utils.data.dataset.Dataset, batch_size, loader_count, **dataloader_kwargs):
    assert batch_size >= 1 and loader_count >= 1
    part_size = len(dset) // loader_count
    # Make even sized batches for the first n-1 loaders, and stick remaining elements in last
    head = (loader_count-1)*(part_size,)
    sizes = *head, len(dset)-sum(head),
    #for i in sizes: print(i)
    # Make a data loader for each dataset
    return [torch.utils.data.DataLoader(i, batch_size=batch_size, shuffle=True, **dataloader_kwargs) for i in torch.utils.data.random_split(dset, sizes, None)] 

def convert_np_torch(item):
    if item in convert_np_torch._info: return convert_np_torch._info[item]
    elif item == np.float32: return torch.float32 # type: ignore
    elif item == np.int8: return torch.int8 # type: ignore
    elif item == np.int16: return torch.int16 # type: ignore
    print(f"Dtype ({item}) not implemented.")
    assert 0 and "Dtype not implemented."

convert_np_torch._info = {
        np.bool       : torch.bool, # type: ignore
        np.uint8      : torch.uint8, # type: ignore
        np.int8       : torch.int8, # type: ignore
        np.int16      : torch.int16, # type: ignore
        np.int32      : torch.int32, # type: ignore 
        np.int64      : torch.int64, # type: ignore
        np.float16    : torch.float16, # type: ignore
        np.float32    : torch.float32, # type: ignore
        np.float64    : torch.float64, # type: ignore
        np.complex64  : torch.complex64, # type: ignore
        np.complex128 : torch.complex128} # type: ignore

