import torch

def batch_index_select(tensor, idx, dim):
    assert dim != 0, "dim 0 invalid, this is the batch dim"

    device = tensor.device
    batch_size = tensor.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    if dim == 1:
        return tensor[batch_indices, idx]
    elif dim == 2:
        return tensor[batch_indices, :, idx]
    elif dim == 3:
        return tensor[batch_indices, :, :, idx]
    elif dim == 4:
        return tensor[batch_indices, :, :, :, idx]
    else:
        raise NotImplementedError("Sorry, haven't figured out how to deliver infinite flexibility here.")
